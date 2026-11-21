// this.cu - GPU vanity search for Bitcoin P2PKH addresses ("1...")
//
// BOTH engines match the ADDRESS (the Base58 text). Same pattern syntax, same GPU matcher,
// optional private-key range, live keys/sec readout. The ONLY difference is the odds estimate:
//   Engine 1: plain estimate - every fixed character counts as 1 in 58
//   Engine 2: Base58-aware estimate (default) - uses the real occurrence probability: the first
//             character after the leading 1 is only 2..Q (~1 in 23), later ones are 1 in 58
//
// BUILD
//   nvcc -O3 -arch=native -o this this.cu       (CUDA 11.5+)
//   nvcc -O3 -arch=sm_75  -o this this.cu       (GTX 1660 Ti = Turing, any CUDA)
//   Windows: run from "x64 Native Tools Command Prompt for VS" (nvcc needs the MSVC compiler).
//
// RUN
//   ./this --selftest                      run this first, every line must say PASS
//   ./this                                 asks for everything interactively
//   ./this --pattern "1Kat???xyz"          see PATTERN below
//   ./this --prefix 1Kat --suffix xyz      start / end of the address
//   ./this --prefix 1K --pos 6=A,12=z      fixed characters by 1-based address position
//   ./this --engine 1 --pattern "1Kat*xyz" same search, plain 1-in-58 estimate
//   ./this --pattern "1Kat*" --start 4000000000000000 --end 4fffffffffffffff [--random-start]
//   ./this --pattern "1Kat*" --bits 45,52,58 [--jump-after N]   cycle through bit ranges, next one after each match
// Other options: --count N  --blocks N  --threads N  --iters N  --no-tune  --out FILE  --checkpoint FILE  --resume
//
// PATTERN (--pattern, or typed at the interactive prompt). Case-sensitive, Base58 has no 0 O I l.
//   ?   any one character
//   *   any run of characters (use at most one); the rest is anchored to both ends
//   Exactly 34 characters: every character sits at its own position
//   Shorter, with a '*':   text before * = start, text after * = end         1Kat*xyz
//   Shorter, no '*':       everything up to and including the last '?' is anchored at the start
//                          (position by position); whatever follows that last '?' is anchored at
//                          the END of the address (suffix). With no '?' at all it is a plain start.
//                          1Ka???u???xyz  = 1Ka at 1-3, u at 7, xyz = last 3 characters
//                          1Kat           = start 1Kat
//   The resolved 34-character pattern is printed before the search starts.
//
// HOW MATCHING WORKS
//   * Leading characters  -> numeric window on the top 64 bits of hash160 (cheap).
//   * Trailing characters -> exact: (hash160||checksum) mod 58^k = last k Base58 digits (<= 7).
//   * Any other fixed position -> the GPU decodes all 33 Base58 digits (slower per key).
//   Every GPU hit is re-encoded on the host and compared as a string before printing.
//
// LIMITS: P2PKH only, compressed keys, case-sensitive, 34-character addresses (~96% of keys;
// addresses starting "11" are not supported).
//
// KEY RANGE: with --start/--end the search walks that range (one key per step). Keys from a
// small or well-known range can be guessed by anyone: do not hold real funds on them.
//
// UNTESTED on a real GPU by the author. Run --selftest first and verify any hit
// independently (import the WIF into a trusted wallet and compare the address).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>
#include <csignal>
#include <map>
#ifndef VANITY_HOST_TEST
#include <cuda_runtime.h>
#endif

typedef unsigned long long u64;
typedef unsigned int u32;

#define BATCH 128          // points per batch-inversion; tune 64..256 (bigger = cheaper inversions, more local memory)
#define NEGFLAG (1ULL << 63)   // hit offset flag: the match is the NEGATED key (n - k)
#define MAXHITS 64
#define ALEN 34            // address length handled by the matcher

#ifndef VANITY_HOST_TEST
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)
#endif

struct fe  { u64 v[4]; };                 // little-endian 4x64 limbs, mod p
struct pt  { fe x, y; };
struct Hit { u64 off; u32 h[5]; };

// ------------------------------------------------------------------ constants
#define CP 0x1000003D1ULL                 // 2^256 mod p
__device__ __constant__ u64 c_lo64, c_hi64, c_sufMod, c_sufVal, c_sufMu;   // c_sufMu = floor((2^64-1)/c_sufMod)
__device__ __constant__ u32 c_min33[6];               // 58^32 as 6 big-endian words: ws >= this <=> 34-character address
__device__ __constant__ int c_lastGroup;              // highest 5-digit group that holds a fixed digit (0..6)
__device__ __constant__ int c_hasPrefix, c_hasSuffix, c_needFull, c_range;
__device__ __constant__ signed char c_pat[ALEN];   // Base58 index wanted at each address position, -1 = any
__device__ u64 d_acc, d_acc2; __device__ int d_is34;
__device__ unsigned char d_dig[33];
__device__ pt d_tab[BATCH + 1];           // d_tab[j] = j*G, j = 1..BATCH (built on the GPU)
__device__ __constant__ pt c_tab[BATCH + 1];   // same table in constant memory: every thread reads the same
                                               // entry at the same time, which the constant cache broadcasts

__device__ __constant__ fe c_G[2] = {
  {{0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL}},
  {{0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL}}
};

// ------------------------------------------------------------ field arithmetic
__device__ __forceinline__ bool fe_ge_p(const fe& a) {
  return a.v[3] == ~0ULL && a.v[2] == ~0ULL && a.v[1] == ~0ULL && a.v[0] >= 0xFFFFFFFEFFFFFC2FULL;
}
__device__ __forceinline__ void add_c(fe& a) {           // a += C (mod 2^256)
  u64 x = a.v[0] + CP; u64 c = x < a.v[0]; a.v[0] = x;
  #pragma unroll
  for (int i = 1; i < 4; i++) { x = a.v[i] + c; c = x < a.v[i]; a.v[i] = x; }
}
__device__ __forceinline__ void fe_add(fe& r, const fe& a, const fe& b) {
  u64 c = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    u64 x = a.v[i] + b.v[i]; u64 c1 = x < a.v[i];
    u64 y = x + c; c1 += y < x; r.v[i] = y; c = c1;
  }
  if (c || fe_ge_p(r)) add_c(r);
}
__device__ __forceinline__ void fe_sub(fe& r, const fe& a, const fe& b) {
  u64 br = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    u64 t = a.v[i] - b.v[i]; u64 b1 = a.v[i] < b.v[i];
    u64 t2 = t - br; b1 += t < br; r.v[i] = t2; br = b1;
  }
  if (br) {                                               // add p == subtract C mod 2^256
    u64 x = r.v[0] - CP; u64 b2 = r.v[0] < CP; r.v[0] = x;
    #pragma unroll
    for (int i = 1; i < 4; i++) { x = r.v[i] - b2; b2 = r.v[i] < b2; r.v[i] = x; }
  }
}
// t[0..5] += m * b  (one row of the schoolbook product). PTX carry chains on the GPU,
// portable C elsewhere (host unit tests).
__device__ __forceinline__ void mad_row(u64 m, const fe& b, u64& t0, u64& t1, u64& t2, u64& t3, u64& t4, u64& t5) {
#if defined(__CUDA_ARCH__)
  asm("mad.lo.cc.u64  %0, %6, %7,  %0;\n\t"
      "madc.lo.cc.u64 %1, %6, %8,  %1;\n\t"
      "madc.lo.cc.u64 %2, %6, %9,  %2;\n\t"
      "madc.lo.cc.u64 %3, %6, %10, %3;\n\t"
      "addc.u64       %4, %4, 0;\n\t"
      "mad.hi.cc.u64  %1, %6, %7,  %1;\n\t"
      "madc.hi.cc.u64 %2, %6, %8,  %2;\n\t"
      "madc.hi.cc.u64 %3, %6, %9,  %3;\n\t"
      "madc.hi.cc.u64 %4, %6, %10, %4;\n\t"
      "addc.u64       %5, %5, 0;\n\t"
      : "+l"(t0), "+l"(t1), "+l"(t2), "+l"(t3), "+l"(t4), "+l"(t5)
      : "l"(m), "l"(b.v[0]), "l"(b.v[1]), "l"(b.v[2]), "l"(b.v[3]));
#else
  u64 tt[6] = {t0, t1, t2, t3, t4, t5}; u64 carry = 0;
  for (int j = 0; j < 4; j++) {
    u64 lo = m * b.v[j], hi = __umul64hi(m, b.v[j]);
    u64 s = tt[j] + lo; u64 c1 = s < lo;
    u64 s2 = s + carry; c1 += s2 < s;
    tt[j] = s2; carry = hi + c1;
  }
  u64 s = tt[4] + carry; u64 c1 = s < carry; tt[4] = s; tt[5] += c1;
  t0 = tt[0]; t1 = tt[1]; t2 = tt[2]; t3 = tt[3]; t4 = tt[4]; t5 = tt[5];
#endif
}
__device__ __forceinline__ void fe_mul(fe& r, const fe& a, const fe& b) {
  u64 t[9];
  #pragma unroll
  for (int i = 0; i < 9; i++) t[i] = 0;
  mad_row(a.v[0], b, t[0], t[1], t[2], t[3], t[4], t[5]);
  mad_row(a.v[1], b, t[1], t[2], t[3], t[4], t[5], t[6]);
  mad_row(a.v[2], b, t[2], t[3], t[4], t[5], t[6], t[7]);
  mad_row(a.v[3], b, t[3], t[4], t[5], t[6], t[7], t[8]);
  u64 m[5]; u64 carry = 0;                                // reduce: lo + hi*C
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    u64 lo = t[4 + i] * CP, hi = __umul64hi(t[4 + i], CP);
    u64 s = lo + carry; hi += s < lo; m[i] = s; carry = hi;
  }
  m[4] = carry;
  u64 s[5]; u64 c = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    u64 x = t[i] + m[i]; u64 c1 = x < t[i];
    u64 y = x + c; c1 += y < x; s[i] = y; c = c1;
  }
  s[4] = m[4] + c;
  u64 lo = s[4] * CP, hi = __umul64hi(s[4], CP);          // fold the top limb
  u64 x = s[0] + lo; c = x < s[0]; s[0] = x;
  x = s[1] + hi; u64 c1 = x < s[1]; u64 y = x + c; c1 += y < x; s[1] = y; c = c1;
  x = s[2] + c; c = x < s[2]; s[2] = x;
  x = s[3] + c; c = x < s[3]; s[3] = x;
  fe o; o.v[0] = s[0]; o.v[1] = s[1]; o.v[2] = s[2]; o.v[3] = s[3];
  if (c) add_c(o);
  if (fe_ge_p(o)) add_c(o);
  r = o;
}
__device__ __forceinline__ void fe_sqr(fe& r, const fe& a) { fe_mul(r, a, a); }
__device__ void fe_sqrn(fe& r, const fe& a, int n) { r = a; for (int i = 0; i < n; i++) fe_sqr(r, r); }
// a^(p-2) with a short addition chain: 255 squarings + 15 multiplications
// (plain square-and-multiply needed about 500 multiplications).
__device__ void fe_inv(fe& r, const fe& a) {
  fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
  fe_sqr(x2, a);     fe_mul(x2, x2, a);
  fe_sqr(x3, x2);    fe_mul(x3, x3, a);
  fe_sqrn(x6, x3, 3);    fe_mul(x6, x6, x3);
  fe_sqrn(x9, x6, 3);    fe_mul(x9, x9, x3);
  fe_sqrn(x11, x9, 2);   fe_mul(x11, x11, x2);
  fe_sqrn(x22, x11, 11); fe_mul(x22, x22, x11);
  fe_sqrn(x44, x22, 22); fe_mul(x44, x44, x22);
  fe_sqrn(x88, x44, 44); fe_mul(x88, x88, x44);
  fe_sqrn(x176, x88, 88);  fe_mul(x176, x176, x88);
  fe_sqrn(x220, x176, 44); fe_mul(x220, x220, x44);
  fe_sqrn(x223, x220, 3);  fe_mul(x223, x223, x3);
  fe_sqrn(t, x223, 23); fe_mul(t, t, x22);
  fe_sqrn(t, t, 5);     fe_mul(t, t, a);
  fe_sqrn(t, t, 3);     fe_mul(t, t, x2);
  fe_sqrn(t, t, 2);     fe_mul(r, t, a);
}

// ------------------------------------------------------------- scalar multiply
struct jp { fe X, Y, Z; };
__device__ void jp_dbl(jp& p) {                           // a = 0 curve
  fe A, B, C, D, E, F, t;
  fe_sqr(A, p.X); fe_sqr(B, p.Y); fe_sqr(C, B);
  fe_add(t, p.X, B); fe_sqr(t, t); fe_sub(t, t, A); fe_sub(t, t, C); fe_add(D, t, t);
  fe_add(E, A, A); fe_add(E, E, A);
  fe_sqr(F, E);
  fe X3, Y3, Z3;
  fe_sub(X3, F, D); fe_sub(X3, X3, D);
  fe_sub(t, D, X3); fe_mul(Y3, E, t);
  fe_add(C, C, C); fe_add(C, C, C); fe_add(C, C, C);      // 8C
  fe_sub(Y3, Y3, C);
  fe_mul(Z3, p.Y, p.Z); fe_add(Z3, Z3, Z3);
  p.X = X3; p.Y = Y3; p.Z = Z3;
}
__device__ void jp_madd(jp& p, const fe& x2, const fe& y2) {   // p += (x2,y2) affine
  fe Z1Z1, U2, S2, H, HH, I, J, r, V, t, X3, Y3, Z3;
  fe_sqr(Z1Z1, p.Z);
  fe_mul(U2, x2, Z1Z1);
  fe_mul(S2, y2, p.Z); fe_mul(S2, S2, Z1Z1);
  fe_sub(H, U2, p.X);
  fe_sqr(HH, H);
  fe_add(I, HH, HH); fe_add(I, I, I);
  fe_mul(J, H, I);
  fe_sub(r, S2, p.Y); fe_add(r, r, r);
  fe_mul(V, p.X, I);
  fe_sqr(X3, r); fe_sub(X3, X3, J); fe_sub(X3, X3, V); fe_sub(X3, X3, V);
  fe_sub(t, V, X3); fe_mul(Y3, r, t);
  fe_mul(t, p.Y, J); fe_add(t, t, t); fe_sub(Y3, Y3, t);
  fe_add(Z3, p.Z, H); fe_sqr(Z3, Z3); fe_sub(Z3, Z3, Z1Z1); fe_sub(Z3, Z3, HH);
  p.X = X3; p.Y = Y3; p.Z = Z3;
}
__device__ void scalar_mul_G(const u64 k[4], fe& ax, fe& ay) {
  jp p; bool inf = true;
  p.X = p.Y = p.Z = fe{{0, 0, 0, 0}};
  for (int i = 255; i >= 0; i--) {
    if (!inf) jp_dbl(p);
    if ((k[i >> 6] >> (i & 63)) & 1) {
      if (inf) { p.X = c_G[0]; p.Y = c_G[1]; p.Z = fe{{1, 0, 0, 0}}; inf = false; }
      else jp_madd(p, c_G[0], c_G[1]);
    }
  }
  fe zi, zi2, zi3;
  fe_inv(zi, p.Z); fe_sqr(zi2, zi); fe_mul(zi3, zi2, zi);
  fe_mul(ax, p.X, zi2); fe_mul(ay, p.Y, zi3);
}

// -------------------------------------------------------------------- SHA-256
__device__ __forceinline__ u32 rotr(u32 x, int n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ u32 rotl(u32 x, int n) { return __funnelshift_l(x, x, n); }

__device__ __constant__ u32 K256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

// One SHA-256 block (already padded, 16 big-endian words) from the standard IV.
__device__ __forceinline__ void sha256_16(const u32 in[16], u32 out[8]) {
  u32 w[64];
  #pragma unroll
  for (int i = 0; i < 16; i++) w[i] = in[i];
  #pragma unroll
  for (int i = 16; i < 64; i++) {
    u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  u32 a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a,
      e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
  #pragma unroll
  for (int i = 0; i < 64; i++) {
    u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    u32 ch = (e & f) ^ (~e & g);
    u32 t1 = h + S1 + ch + K256[i] + w[i];
    u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    u32 mj = (a & b) ^ (a & c) ^ (b & c);
    u32 t2 = S0 + mj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  out[0] = 0x6a09e667 + a; out[1] = 0xbb67ae85 + b; out[2] = 0x3c6ef372 + c; out[3] = 0xa54ff53a + d;
  out[4] = 0x510e527f + e; out[5] = 0x9b05688c + f; out[6] = 0x1f83d9ab + g; out[7] = 0x5be0cd19 + h;
}

__device__ __forceinline__ void sha256_pub33(const fe& x, u32 parity, u32 out[8]) {
  u32 xw[8];
  #pragma unroll
  for (int i = 0; i < 4; i++) { xw[2 * i] = (u32)(x.v[3 - i] >> 32); xw[2 * i + 1] = (u32)x.v[3 - i]; }
  u32 w[16];
  w[0] = ((0x02u | parity) << 24) | (xw[0] >> 8);
  #pragma unroll
  for (int i = 1; i < 8; i++) w[i] = (xw[i - 1] << 24) | (xw[i] >> 8);
  w[8] = (xw[7] << 24) | 0x00800000u;
  #pragma unroll
  for (int i = 9; i < 15; i++) w[i] = 0;
  w[15] = 33 * 8;
  sha256_16(w, out);
}

// ------------------------------------------------------------------ RIPEMD-160
__device__ __forceinline__ u32 rf(int r, u32 x, u32 y, u32 z) {
  switch (r) {
    case 0: return x ^ y ^ z;
    case 1: return (x & y) | (~x & z);
    case 2: return (x | ~y) ^ z;
    case 3: return (x & z) | (y & ~z);
    default: return x ^ (y | ~z);
  }
}
__device__ __forceinline__ void ripemd160_32(const u32 in[8], u32 out[5]) {
  const unsigned char RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
  const unsigned char RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
  const unsigned char SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
  const unsigned char SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
  const u32 KL[5] = {0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu};
  const u32 KR[5] = {0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u};

  u32 X[16];
  #pragma unroll
  for (int i = 0; i < 8; i++) X[i] = __byte_perm(in[i], 0, 0x0123);   // LE words
  X[8] = 0x80; X[9] = X[10] = X[11] = X[12] = X[13] = 0; X[14] = 256; X[15] = 0;

  u32 al = 0x67452301, bl = 0xEFCDAB89, cl = 0x98BADCFE, dl = 0x10325476, el = 0xC3D2E1F0;
  u32 ar = al, br = bl, cr = cl, dr = dl, er = el;
  #pragma unroll
  for (int j = 0; j < 80; j++) {
    int rd = j >> 4;
    u32 t = rotl(al + rf(rd, bl, cl, dl) + X[RL[j]] + KL[rd], SL[j]) + el;
    al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = t;
    t = rotl(ar + rf(4 - rd, br, cr, dr) + X[RR[j]] + KR[rd], SR[j]) + er;
    ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = t;
  }
  u32 t = 0xEFCDAB89 + cl + dr;
  out[1] = 0x98BADCFE + dl + er;
  out[2] = 0x10325476 + el + ar;
  out[3] = 0xC3D2E1F0 + al + br;
  out[4] = 0x67452301 + bl + cr;
  out[0] = t;
}

__device__ __forceinline__ void hash160(const fe& x, u32 parity, u32 h[5]) {
  u32 s[8]; sha256_pub33(x, parity, s); ripemd160_32(s, h);
}

// ---------------------------------------------------- Base58 address arithmetic
// ws = the 24 bytes (hash160 || 4-byte checksum) as 6 big-endian words. The address is
// '1' followed by ws written in Base58 (33 digits for 34-char addresses).
__device__ __forceinline__ void addr_words(const u32 h[5], u32 ws[6]) {
  u32 hb[5];
  #pragma unroll
  for (int i = 0; i < 5; i++) hb[i] = __byte_perm(h[i], 0, 0x0123);   // big-endian words
  u32 b[16];
  b[0] = hb[0] >> 8;                                      // 0x00 version byte || hash160
  #pragma unroll
  for (int i = 1; i < 5; i++) b[i] = (hb[i - 1] << 24) | (hb[i] >> 8);
  b[5] = (hb[4] << 24) | 0x00800000u;
  #pragma unroll
  for (int i = 6; i < 15; i++) b[i] = 0;
  b[15] = 21 * 8;
  u32 d[8]; sha256_16(b, d);
  u32 c[16];
  #pragma unroll
  for (int i = 0; i < 8; i++) c[i] = d[i];
  c[8] = 0x80000000u;
  #pragma unroll
  for (int i = 9; i < 15; i++) c[i] = 0;
  c[15] = 32 * 8;
  u32 e[8]; sha256_16(c, e);
  ws[0] = hb[0]; ws[1] = hb[1]; ws[2] = hb[2]; ws[3] = hb[3]; ws[4] = hb[4]; ws[5] = e[0];
}
// ws mod M. With M = 58^k this is the numeric value of the last k Base58 digits.
__device__ __forceinline__ u64 words_mod(const u32 ws[6], u64 M) {
  u64 acc = 0;
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    acc = ((acc << 16) | (ws[i] >> 16)) % M;                  // M < 2^46 so this fits in u64
    acc = ((acc << 16) | (ws[i] & 0xFFFFu)) % M;
  }
  return acc;
}
__device__ __forceinline__ u64 addr_mod(const u32 h[5], u64 M) { u32 ws[6]; addr_words(h, ws); return words_mod(ws, M); }

// x mod M without a 64-bit division: mu = floor((2^64-1)/M), x < 2^62. The quotient estimate is off by at most 1.
__device__ __forceinline__ u64 mod_fast(u64 x, u64 M, u64 mu) {
  u64 q = __umul64hi(x, mu);
  u64 r = x - q * M;
  if (r >= M) r -= M;
  if (r >= M) r -= M;
  return r;
}
// Same result as words_mod, about 3 instructions per step instead of a 64-bit division.
__device__ __forceinline__ u64 words_mod_fast(const u32 ws[6], u64 M, u64 mu) {
  u64 acc = 0;
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    acc = mod_fast((acc << 16) | (ws[i] >> 16), M, mu);
    acc = mod_fast((acc << 16) | (ws[i] & 0xFFFFu), M, mu);
  }
  return acc;
}
// A 34-character address has a 33-digit Base58 body, i.e. ws >= 58^32 (replaces decoding the top digit).
__device__ __forceinline__ bool ws_is_34(const u32 ws[6]) {
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    if (ws[i] > c_min33[i]) return true;
    if (ws[i] < c_min33[i]) return false;
  }
  return true;
}
// Decodes 5 Base58 digits at a time (least significant first) and rejects at the first wrong fixed
// digit; stops once no fixed digit is left above the current group.
__device__ __forceinline__ bool match_digits(const u32 ws[6]) {
  u32 n[6];
  #pragma unroll
  for (int i = 0; i < 6; i++) n[i] = ws[i];
  #pragma unroll
  for (int g = 0; g < 6; g++) {
    u64 rem = 0;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
      u64 cur = (rem << 32) | n[i];
      n[i] = (u32)(cur / 656356768ULL);                       // 58^5
      rem = cur % 656356768ULL;
    }
    u32 r = (u32)rem;
    #pragma unroll
    for (int d = 0; d < 5; d++) {
      int want = c_pat[ALEN - 1 - (5 * g + d)];
      u32 dig = r % 58u; r /= 58u;
      if (want >= 0 && dig != (u32)want) return false;
    }
    if (g >= c_lastGroup) return true;                        // nothing fixed further up
  }
  u32 r = n[5];                                               // top three digits (< 58^3)
  #pragma unroll
  for (int d = 0; d < 3; d++) {
    int want = c_pat[ALEN - 1 - (30 + d)];
    u32 dig = r % 58u; r /= 58u;
    if (want >= 0 && dig != (u32)want) return false;
  }
  return true;
}

// All 33 Base58 digits of ws, dg[0] = least significant. Divides by 58^5 six times.
__device__ __forceinline__ void b58_digits(const u32 ws[6], unsigned char dg[33]) {
  u32 n[6];
  #pragma unroll
  for (int i = 0; i < 6; i++) n[i] = ws[i];
  #pragma unroll
  for (int g = 0; g < 6; g++) {
    u64 rem = 0;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
      u64 cur = (rem << 32) | n[i];
      n[i] = (u32)(cur / 656356768ULL);                       // 58^5
      rem = cur % 656356768ULL;
    }
    u32 r = (u32)rem;
    #pragma unroll
    for (int d = 0; d < 5; d++) { dg[5 * g + d] = (unsigned char)(r % 58u); r /= 58u; }
  }
  u32 r = n[5];                                               // < 58^3 here
  dg[30] = (unsigned char)(r % 58u); r /= 58u;
  dg[31] = (unsigned char)(r % 58u); r /= 58u;
  dg[32] = (unsigned char)r;
}

// Does the compressed pubkey (parity||x) give an address that satisfies the pattern? h receives its hash160.
__device__ __forceinline__ bool test(const fe& x, u32 parity, u32 h[5]) {
  hash160(x, parity, h);
  if (c_hasPrefix) {
    u64 top = ((u64)__byte_perm(h[0], 0, 0x0123) << 32) | __byte_perm(h[1], 0, 0x0123);
    if (top < c_lo64 || top > c_hi64) return false;
  }
  if (c_hasSuffix | c_needFull) {
    u32 ws[6]; addr_words(h, ws);
    if (c_hasSuffix && words_mod_fast(ws, c_sufMod, c_sufMu) != c_sufVal) return false;
    if (c_needFull) {
      if (!ws_is_34(ws)) return false;                        // not a 34-character address
      if (!match_digits(ws)) return false;
    }
  }
  return true;
}
__device__ __forceinline__ void report(u64 off, const u32 h[5], Hit* hits, u32* cnt) {
  u32 i = atomicAdd(cnt, 1u);
  if (i < MAXHITS) { hits[i].off = off; for (int k = 0; k < 5; k++) hits[i].h[k] = h[k]; }
}
// Negation trick: P=(x,y) and -P=(x,-y) share x, so one x-coordinate tests two keys (k and n-k).
// yp is the y parity of P (key k).
__device__ __forceinline__ void check2(const fe& x, u32 yp, u64 off, Hit* hits, u32* cnt) {
  u32 h0[5], h1[5];
  bool m0 = test(x, 0, h0), m1 = test(x, 1, h1);
  if (m0) report(off | (yp != 0 ? NEGFLAG : 0ULL), h0, hits, cnt);
  if (m1) report(off | (yp != 1 ? NEGFLAG : 0ULL), h1, hits, cnt);
}
// Range mode: only key k itself is wanted (n-k is outside the range).
__device__ __forceinline__ void check1(const fe& x, u32 yp, u64 off, Hit* hits, u32* cnt) {
  u32 h[5];
  if (test(x, yp, h)) report(off, h, hits, cnt);
}

// -------------------------------------------------------------------- kernels
__global__ void build_table() {
  int j = threadIdx.x + 1;                                // 1..BATCH
  u64 k[4] = {(u64)j, 0, 0, 0};
  scalar_mul_G(k, d_tab[j].x, d_tab[j].y);
}

__global__ void init_points(fe base, u64 stride, fe* px, fe* py) {
  u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  u64 off = (u64)t * stride, k[4]; u64 c = 0;
  u64 x = base.v[0] + off; c = x < base.v[0]; k[0] = x;
  #pragma unroll
  for (int i = 1; i < 4; i++) { x = base.v[i] + c; c = x < base.v[i]; k[i] = x; }
  scalar_mul_G(k, px[t], py[t]);
}

__global__ void __launch_bounds__(128) search(const fe* px_, const fe* py_, int iters, u64 stride,
                                              Hit* hits, u32* cnt) {
  u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  fe px = px_[t], py = py_[t];
  u64 tOff = (u64)t * stride;
  fe pre[BATCH];                                          // prefix products; dx is recomputed (saves local memory)

  for (int it = 0; it < iters; it++) {
    u64 off0 = tOff + (u64)it * BATCH;
    if (c_range) check1(px, (u32)(py.v[0] & 1), off0, hits, cnt);      // j = 0
    else         check2(px, (u32)(py.v[0] & 1), off0, hits, cnt);

    {                                                     // Montgomery batch inversion, forward pass
      fe d; fe_sub(d, c_tab[1].x, px); pre[0] = d;
      for (int i = 1; i < BATCH; i++) { fe_sub(d, c_tab[i + 1].x, px); fe_mul(pre[i], pre[i - 1], d); }
    }
    fe inv; fe_inv(inv, pre[BATCH - 1]);

    fe nx = px, ny = py;
    for (int i = BATCH - 1; i >= 0; i--) {
      fe di, dxi;
      fe_sub(dxi, c_tab[i + 1].x, px);
      if (i > 0) { fe_mul(di, inv, pre[i - 1]); fe_mul(inv, inv, dxi); } else di = inv;
      int j = i + 1;
      fe s, x3, y3, tmp;
      fe_sub(tmp, c_tab[j].y, py); fe_mul(s, tmp, di);
      fe_sqr(x3, s); fe_sub(x3, x3, px); fe_sub(x3, x3, c_tab[j].x);
      if (j < BATCH) {
        if (c_range) {                                    // need y to know which pubkey parity is key k
          fe_sub(tmp, px, x3); fe_mul(y3, s, tmp); fe_sub(y3, y3, py);
          check1(x3, (u32)(y3.v[0] & 1), off0 + j, hits, cnt);
        } else {                                          // y only needed when something matches
          u32 h0[5], h1[5];
          bool m0 = test(x3, 0, h0), m1 = test(x3, 1, h1);
          if (m0 | m1) {
            fe_sub(tmp, px, x3); fe_mul(y3, s, tmp); fe_sub(y3, y3, py);
            u32 yp = (u32)(y3.v[0] & 1);
            if (m0) report((off0 + j) | (yp != 0 ? NEGFLAG : 0ULL), h0, hits, cnt);
            if (m1) report((off0 + j) | (yp != 1 ? NEGFLAG : 0ULL), h1, hits, cnt);
          }
        }
      } else {                                            // j == BATCH: next starting point, needs y
        fe_sub(tmp, px, x3); fe_mul(y3, s, tmp); fe_sub(y3, y3, py);
        nx = x3; ny = y3;
      }
    }
    px = nx; py = ny;
  }
}

__global__ void selftest(u64 M4, u64 mu4) {
  u32 h[5]; hash160(c_G[0], 0, h);
  const unsigned char want[20] = {0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
  bool okH = true;
  for (int i = 0; i < 20; i++) okH &= (((h[i >> 2] >> (8 * (i & 3))) & 0xff) == want[i]);
  printf("hash160(G)            : %s\n", okH ? "PASS" : "FAIL");
  u64 k2[4] = {2, 0, 0, 0}, k6[4] = {6, 0, 0, 0}; fe x2, y2, x6, y6;
  scalar_mul_G(k2, x2, y2); scalar_mul_G(k6, x6, y6);
  bool ok2 = x2.v[3] == 0xc6047f9441ed7d6dULL && x2.v[2] == 0x3045406e95c07cd8ULL &&
             x2.v[1] == 0x5c778e4b8cef3ca7ULL && x2.v[0] == 0xabac09b95c709ee5ULL;
  printf("2G x-coordinate       : %s\n", ok2 ? "PASS" : "FAIL");
  fe s, x3, y3, tmp, di;                                  // G + 5G must equal 6G
  fe_sub(tmp, d_tab[5].x, c_G[0]); fe_inv(di, tmp);
  fe_sub(tmp, d_tab[5].y, c_G[1]); fe_mul(s, tmp, di);
  fe_sqr(x3, s); fe_sub(x3, x3, c_G[0]); fe_sub(x3, x3, d_tab[5].x);
  fe_sub(tmp, c_G[0], x3); fe_mul(y3, s, tmp); fe_sub(y3, y3, c_G[1]);
  bool ok = true;
  for (int i = 0; i < 4; i++) ok &= (x3.v[i] == x6.v[i]) && (y3.v[i] == y6.v[i]);
  printf("batch add G+5G == 6G  : %s\n", ok ? "PASS" : "FAIL");
  u32 ws[6]; addr_words(h, ws);
  d_acc = words_mod(ws, M4);
  d_acc2 = words_mod_fast(ws, M4, mu4);
  d_is34 = ws_is_34(ws) ? 1 : 0;
  unsigned char dg[33]; b58_digits(ws, dg);
  for (int i = 0; i < 33; i++) d_dig[i] = dg[i];
}

__global__ void selftest_digits(int expect) {             // c_pat / c_lastGroup were set by the host
  u32 h[5]; hash160(c_G[0], 0, h);
  u32 ws[6]; addr_words(h, ws);
  bool got = match_digits(ws);
  printf("digit matcher (%s)%s: %s\n", expect ? "should match" : "should reject", expect ? "  " : "  ", got == (expect != 0) ? "PASS" : "FAIL");
}

// ----------------------------------------------------------------------- host
static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static int b58idx(char c) { const char* p = strchr(B58, c); return (p && c) ? (int)(p - B58) : -1; }

static std::string b58enc(const unsigned char* d, size_t n) {
  size_t z = 0; while (z < n && d[z] == 0) z++;
  std::vector<unsigned char> t(d, d + n); std::string out; size_t start = z;
  while (start < n) {
    int rem = 0;
    for (size_t i = start; i < n; i++) { int cur = rem * 256 + t[i]; t[i] = (unsigned char)(cur / 58); rem = cur % 58; }
    out.push_back(B58[rem]);
    while (start < n && t[start] == 0) start++;
  }
  out.append(z, '1'); std::reverse(out.begin(), out.end()); return out;
}

static void sha256_host(const unsigned char* d, size_t n, unsigned char out[32]) {
  static const u32 K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  u32 H[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::vector<unsigned char> m(d, d + n); m.push_back(0x80);
  while (m.size() % 64 != 56) m.push_back(0);
  u64 bits = (u64)n * 8; for (int i = 7; i >= 0; i--) m.push_back((unsigned char)(bits >> (8 * i)));
  auto ror = [](u32 x, int r) { return (x >> r) | (x << (32 - r)); };
  for (size_t off = 0; off < m.size(); off += 64) {
    u32 w[64];
    for (int i = 0; i < 16; i++) w[i] = (u32)m[off+4*i] << 24 | (u32)m[off+4*i+1] << 16 | (u32)m[off+4*i+2] << 8 | m[off+4*i+3];
    for (int i = 16; i < 64; i++) {
      u32 s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3), s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a=H[0],b=H[1],c=H[2],dd=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    for (int i = 0; i < 64; i++) {
      u32 t1 = h + (ror(e,6)^ror(e,11)^ror(e,25)) + ((e&f)^(~e&g)) + K[i] + w[i];
      u32 t2 = (ror(a,2)^ror(a,13)^ror(a,22)) + ((a&b)^(a&c)^(b&c));
      h=g; g=f; f=e; e=dd+t1; dd=c; c=b; b=a; a=t1+t2;
    }
    H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=dd;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
  }
  for (int i = 0; i < 8; i++) for (int j = 0; j < 4; j++) out[4*i+j] = (unsigned char)(H[i] >> (24 - 8*j));
}

// Base58Check: payload || first 4 bytes of SHA256d(payload)
static std::string b58check(const std::vector<unsigned char>& payload) {
  unsigned char a[32], b[32]; sha256_host(payload.data(), payload.size(), a); sha256_host(a, 32, b);
  std::vector<unsigned char> v = payload; v.insert(v.end(), b, b + 4);
  return b58enc(v.data(), v.size());
}

typedef std::array<unsigned char, 32> big;              // 256-bit big-endian
static void big_mul(big& a, unsigned m) { unsigned c = 0; for (int i = 31; i >= 0; i--) { unsigned cur = a[i] * m + c; a[i] = cur & 0xFF; c = cur >> 8; } }
static void big_add(big& a, unsigned v) { unsigned c = v; for (int i = 31; i >= 0 && c; i--) { unsigned cur = a[i] + c; a[i] = cur & 0xFF; c = cur >> 8; } }
static void big_sub1(big& a) { for (int i = 31; i >= 0; i--) { if (a[i]--) break; } }
static u64 top64(const big& a) { u64 r = 0; for (int i = 8; i < 16; i++) r = (r << 8) | a[i]; return r; }

static void big_sub(big& a, const big& b) { int br = 0; for (int i = 31; i >= 0; i--) { int t = (int)a[i] - (int)b[i] - br; br = t < 0; a[i] = (unsigned char)(t & 0xFF); } }
static double frac192(const big& a) { double r = 0; for (int i = 8; i < 32; i++) r = r * 256.0 + a[i]; return ldexp(r, -192); }

// Leading digits (after the '1') -> window on the top 64 bits of hash160, for 34-char addresses.
// *prob (optional) = exact probability that a random key's address falls in that window and has 34 chars.
static bool prefix_window(const std::string& digits, u64& lo64, u64& hi64, double* prob = nullptr) {
  int m1 = (int)digits.size(), r = 33 - m1;
  big val{}; for (char ch : digits) { big_mul(val, 58); big_add(val, (unsigned)b58idx(ch)); }
  big lo = val, hi = val; big_add(hi, 1);
  for (int i = 0; i < r; i++) { big_mul(lo, 58); big_mul(hi, 58); }
  big_sub1(hi);
  big mn{}; mn[31] = 1; for (int i = 0; i < 32; i++) big_mul(mn, 58);     // 58^32
  big mx{}; for (int i = 8; i < 32; i++) mx[i] = 0xFF;                    // 2^192 - 1
  if (memcmp(lo.data(), mn.data(), 32) < 0) lo = mn;
  if (memcmp(hi.data(), mx.data(), 32) > 0) hi = mx;
  if (memcmp(lo.data(), hi.data(), 32) > 0) return false;
  if (prob) { big w = hi; big_sub(w, lo); big_add(w, 1); *prob = frac192(w); }
  lo64 = top64(lo); hi64 = top64(hi); return true;
}

static u64 b58val(const std::string& s) { u64 v = 0; for (char c : s) v = v * 58 + b58idx(c); return v; }
static u64 pow58(int k) { u64 v = 1; for (int i = 0; i < k; i++) v *= 58; return v; }
static void print_be(const u64 v[4]) { printf("%016llx%016llx%016llx%016llx", v[3], v[2], v[1], v[0]); }

// ------------------------------------------------- address patterns (both engines)
struct Pattern { char c[ALEN]; };                       // Base58 address, 0 = any character

// Expands --pattern into exactly ALEN characters ('?' = any). See the PATTERN section at the top.
static bool expand_pattern(const std::string& pattern, std::string& e, std::string& err) {
  size_t stars = (size_t)std::count(pattern.begin(), pattern.end(), '*');
  if (stars > 1) { err = "use at most one '*' in the pattern"; return false; }
  if (pattern.empty()) { e.clear(); return true; }
  if (stars == 1) {
    size_t fixedLen = pattern.size() - 1, at = pattern.find('*');
    if ((int)fixedLen > ALEN) { err = "pattern is longer than " + std::to_string(ALEN) + " characters"; return false; }
    e = pattern.substr(0, at) + std::string(ALEN - fixedLen, '?') + pattern.substr(at + 1);
    return true;
  }
  if ((int)pattern.size() > ALEN) { err = "pattern is longer than " + std::to_string(ALEN) + " characters"; return false; }
  if ((int)pattern.size() == ALEN) { e = pattern; return true; }
  // Shorter, no '*': head (up to the last '?') anchored at the start, the rest anchored at the end.
  size_t lq = pattern.rfind('?');
  std::string head = lq == std::string::npos ? pattern : pattern.substr(0, lq + 1);
  std::string tail = lq == std::string::npos ? std::string() : pattern.substr(lq + 1);
  e = head + std::string(ALEN - head.size() - tail.size(), '?') + tail;
  return true;
}

// Combines prefix, suffix, pattern and a position list "6=A,12=z" (1-based) into one array of
// ALEN characters (0 = any). Conflicting characters are rejected.
static bool build_pattern(const std::string& prefix, const std::string& suffix, const std::string& pattern,
                          const std::string& pos, Pattern& p, std::string& err) {
  char* c = p.c; const int L = ALEN;
  memset(c, 0, L);
  auto set = [&](int idx, char ch) -> bool {
    if (b58idx(ch) < 0) {
      err = std::string("'") + ch + "' is not a Base58 character (0, O, I and l are not used)"; return false;
    }
    if (c[idx] && c[idx] != ch) { err = "conflicting characters at position " + std::to_string(idx + 1); return false; }
    c[idx] = ch; return true;
  };
  if ((int)prefix.size() > L) { err = "start is longer than the address"; return false; }
  for (size_t i = 0; i < prefix.size(); i++) if (!set((int)i, prefix[i])) return false;
  if ((int)suffix.size() > L) { err = "end is longer than the address"; return false; }
  for (size_t i = 0; i < suffix.size(); i++) if (!set((int)(L - suffix.size() + i), suffix[i])) return false;
  if (!pattern.empty()) {
    std::string e; if (!expand_pattern(pattern, e, err)) return false;
    for (int i = 0; i < L; i++) if (e[i] != '?' && !set(i, e[i])) return false;
  }
  if (!pos.empty()) {
    size_t i = 0;
    while (i < pos.size()) {
      while (i < pos.size() && (pos[i] == ',' || pos[i] == ' ' || pos[i] == ';')) i++;
      if (i >= pos.size()) break;
      size_t j = i; while (j < pos.size() && pos[j] != ',' && pos[j] != ' ' && pos[j] != ';') j++;
      std::string tok = pos.substr(i, j - i); i = j;
      size_t eq = tok.find('=');
      if (eq == std::string::npos || eq == 0 || tok.size() != eq + 2) { err = "bad position entry '" + tok + "' (use N=C, e.g. 6=A)"; return false; }
      int n = atoi(tok.substr(0, eq).c_str());
      if (n < 1 || n > L) { err = "position must be between 1 and " + std::to_string(L); return false; }
      if (!set(n - 1, tok[eq + 1])) return false;
    }
  }
  if (p.c[0] && p.c[0] != '1') { err = "P2PKH addresses start with '1'"; return false; }
  p.c[0] = '1';
  return true;
}

// 58^32 as six big-endian 32-bit words (smallest 33-digit Base58 body = smallest 34-character address)
static void min33_words(u32 out[6]) {
  u32 le[6] = {1, 0, 0, 0, 0, 0};
  for (int r = 0; r < 32; r++) {
    u64 carry = 0;
    for (int i = 0; i < 6; i++) { u64 v = (u64)le[i] * 58 + carry; le[i] = (u32)v; carry = v >> 32; }
  }
  for (int i = 0; i < 6; i++) out[i] = le[5 - i];
}
// Highest 5-digit group (counted from the end of the address) that contains a fixed character.
static int last_group(const signed char* pat) {
  int g = 0;
  for (int i = 1; i < ALEN; i++) if (pat[i] >= 0) { int k = ALEN - 1 - i; g = std::max(g, k >= 30 ? 6 : k / 5); }
  return g;
}

struct PatInfo { int m1, k, fixed; bool hasPrefix, hasSuffix, needFull; };
static PatInfo analyze(const Pattern& p) {
  PatInfo r; int lead = 1; while (lead < ALEN && p.c[lead]) lead++;
  int trail = 0; while (trail < ALEN - 1 && p.c[ALEN - 1 - trail]) trail++;
  r.m1 = std::min(lead - 1, 10);                          // digits used in the prefix window
  r.k = std::min(trail, 7);                               // digits used in the suffix test
  r.hasPrefix = r.m1 >= 1; r.hasSuffix = r.k >= 1;
  r.needFull = false; r.fixed = 0;
  for (int i = 1; i < ALEN; i++) if (p.c[i]) {
    r.fixed++;
    if (!(i <= r.m1 || i >= ALEN - r.k)) r.needFull = true;
  }
  return r;
}
static bool pattern_match(const std::string& addr, const Pattern& p) {
  if (addr.size() != ALEN) return false;
  for (int i = 0; i < ALEN; i++) if (p.c[i] && addr[i] != p.c[i]) return false;
  return true;
}

// ------------------------------------------------------- 256-bit helpers (host)
static const u64 ORDER_N[4]  = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
static const u64 NMINUS1[4]  = {0xBFD25E8CD0364140ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};

static int cmp256(const u64 a[4], const u64 b[4]) {
  for (int i = 3; i >= 0; i--) { if (a[i] < b[i]) return -1; if (a[i] > b[i]) return 1; }
  return 0;
}
static bool is_zero256(const u64 a[4]) { return !(a[0] | a[1] | a[2] | a[3]); }
static u64 add256_u64(u64 a[4], u64 v) {                // a += v, returns carry out
  u64 c = v; for (int i = 0; i < 4; i++) { u64 x = a[i] + c; c = x < a[i]; a[i] = x; } return c;
}
static u64 add256(u64 r[4], const u64 a[4], const u64 b[4]) {   // r = a + b, returns carry out
  u64 c = 0; for (int i = 0; i < 4; i++) {
    u64 x = a[i] + b[i]; u64 c1 = x < a[i]; u64 y = x + c; c1 += y < x; r[i] = y; c = c1; }
  return c;
}
static void sub256(u64 r[4], const u64 a[4], const u64 b[4]) {   // r = a - b (wraps if a < b)
  u64 br = 0; for (int i = 0; i < 4; i++) {
    u64 t = a[i] - b[i]; u64 b1 = a[i] < b[i]; u64 t2 = t - br; b1 += t < br; r[i] = t2; br = b1; }
}
static int bitlen256(const u64 a[4]) {
  for (int i = 3; i >= 0; i--) if (a[i]) { int b = 64; u64 x = a[i]; while (!(x >> 63)) { x <<= 1; b--; } return 64 * i + b; }
  return 0;
}
static double to_double256(const u64 a[4]) {
  double r = 0; for (int i = 3; i >= 0; i--) r = r * 18446744073709551616.0 + (double)a[i]; return r;
}
static bool parse_hex256(std::string s, u64 out[4]) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t st = 0; while (st < s.size() && s[st] == ' ') st++;
  s = s.substr(st);
  if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
  if (s.empty() || s.size() > 64) return false;
  memset(out, 0, 4 * sizeof(u64));
  for (size_t i = 0; i < s.size(); i++) {
    char ch = s[s.size() - 1 - i]; int v;
    if (ch >= '0' && ch <= '9') v = ch - '0'; else if (ch >= 'a' && ch <= 'f') v = ch - 'a' + 10;
    else if (ch >= 'A' && ch <= 'F') v = ch - 'A' + 10; else return false;
    out[i / 16] |= (u64)v << (4 * (i % 16));
  }
  return true;
}
static void rand256_upto(const u64 diff[4], u64 out[4]) {       // uniform in [0, diff]
  std::random_device rd; int bits = bitlen256(diff);
  for (;;) {
    for (int i = 0; i < 4; i++) {
      int n = std::max(0, std::min(64, bits - 64 * i));
      u64 v = ((u64)rd() << 32) | rd();
      out[i] = n == 0 ? 0 : (n == 64 ? v : (v & ((1ULL << n) - 1)));
    }
    if (cmp256(out, diff) <= 0) return;
  }
}

static std::string fmt_rate(double r) {
  char b[64];
  if (r >= 1e9) snprintf(b, sizeof b, "%.2f Gkeys/s", r / 1e9);
  else if (r >= 1e6) snprintf(b, sizeof b, "%.2f Mkeys/s", r / 1e6);
  else if (r >= 1e3) snprintf(b, sizeof b, "%.2f Kkeys/s", r / 1e3);
  else snprintf(b, sizeof b, "%.0f keys/s", r);
  return b;
}
static std::string fmt_dur(double s) {
  char b[64];
  if (!(s < 1e12)) return "n/a";
  if (s < 90) snprintf(b, sizeof b, "%.0fs", s);
  else if (s < 5400) snprintf(b, sizeof b, "%.1fmin", s / 60);
  else if (s < 86400 * 2) snprintf(b, sizeof b, "%.1fh", s / 3600);
  else if (s < 86400 * 730) snprintf(b, sizeof b, "%.1fd", s / 86400);
  else snprintf(b, sizeof b, "%.1fy", s / 86400 / 365);
  return b;
}

// ----------------------------------------------- last-key reporting and checkpoint file
static volatile sig_atomic_t g_stop = 0;
struct RunState {
  bool active = false, haveLast = false, done = false, range = false;
  u64 rs[4] = {0, 0, 0, 0}, re[4] = {0, 0, 0, 0}, next[4] = {0, 0, 0, 0}, last[4] = {0, 0, 0, 0};
  double scanned = 0; int found = 0, engine = 2;
  std::string ckpt, pattern;
  std::string bits; int ridx = 0, jumpAfter = 1;           // bit-range schedule (--bits)
  std::vector<std::string> rnext; std::vector<int> rdone; std::vector<double> rscan;
};
static RunState g_st;
static std::string hex256(const u64 v[4]) {
  char b[65]; snprintf(b, sizeof b, "%016llx%016llx%016llx%016llx", v[3], v[2], v[1], v[0]); return b;
}
static std::string hex_short(const u64 v[4]) {          // without leading zeros
  std::string s = hex256(v); size_t i = 0; while (i + 1 < s.size() && s[i] == '0') i++; return s.substr(i);
}
static void write_checkpoint() {
  if (!g_st.active || g_st.ckpt.empty()) return;
  std::string tmp = g_st.ckpt + ".tmp";
  FILE* f = fopen(tmp.c_str(), "w"); if (!f) return;
  fprintf(f, "# vanity160 checkpoint - continue with: --resume\n");
  fprintf(f, "engine=%d\npattern=%s\nmode=%s\n", g_st.engine, g_st.pattern.c_str(), g_st.range ? "range" : "random");
  fprintf(f, "start=%s\nend=%s\n", hex256(g_st.rs).c_str(), hex256(g_st.re).c_str());
  fprintf(f, "last=%s\nnext=%s\n", g_st.haveLast ? hex256(g_st.last).c_str() : "none", hex256(g_st.next).c_str());
  fprintf(f, "scanned=%.0f\nfound=%d\ndone=%d\n", g_st.scanned, g_st.found, g_st.done ? 1 : 0);
  if (!g_st.bits.empty()) {
    fprintf(f, "bits=%s\njump=%d\nridx=%d\n", g_st.bits.c_str(), g_st.jumpAfter, g_st.ridx);
    for (size_t i = 0; i < g_st.rdone.size(); i++) {
      bool cur = (int)i == g_st.ridx;
      fprintf(f, "rnext_%zu=%s\nrdone_%zu=%d\nrscan_%zu=%.0f\n", i, cur ? hex256(g_st.next).c_str() : g_st.rnext[i].c_str(),
              i, g_st.rdone[i], i, cur ? g_st.scanned : g_st.rscan[i]);
    }
  }
  fclose(f);
  remove(g_st.ckpt.c_str());                            // Windows rename() will not overwrite
  rename(tmp.c_str(), g_st.ckpt.c_str());
}
static bool read_kv(const std::string& path, std::map<std::string, std::string>& kv) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) f = fopen((path + ".tmp").c_str(), "r");      // crash between remove() and rename()
  if (!f) return false;
  char line[512];
  while (fgets(line, sizeof line, f)) {
    std::string s = line;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    size_t eq = s.find('=');
    if (s.empty() || s[0] == '#' || eq == std::string::npos) continue;
    kv[s.substr(0, eq)] = s.substr(eq + 1);
  }
  fclose(f); return true;
}
// Runs at every normal exit, after Ctrl+C / kill / terminal close (signal handlers only set a
// flag) and after a CUDA error (CK calls exit). A power cut or kill -9 cannot be caught: use
// the checkpoint file, which is rewritten about once a second.
static void report_last_key() {
  static bool once = false; if (once) return; once = true;
  if (!g_st.active) return;
  write_checkpoint();
  printf("\n");
  if (g_st.haveLast) {
    printf("Last key scanned : %s\n", hex256(g_st.last).c_str());
    printf("Next key to scan : %s\n", hex256(g_st.next).c_str());
    if (!g_st.range) printf("                   (random mode: the negated keys n-k were tested too)\n");
    if (!g_st.bits.empty()) printf("                   (bit-range schedule %s: these keys belong to range #%d)\n", g_st.bits.c_str(), g_st.ridx + 1);
    if (!g_st.done) printf("Resume with      : --resume --checkpoint %s\n", g_st.ckpt.c_str());
  } else printf("No complete batch was scanned before exit.\n");
  fflush(stdout);
}
static void on_signal(int) { g_stop = 1; }

#ifndef VANITY_HOST_TEST
static void usage() {
  printf(
"vanity160 - GPU Bitcoin P2PKH vanity address search (compressed keys, 34-character '1...' addresses)\n"
"\n"
"USAGE\n"
"  this [options]            no pattern given = interactive questions\n"
"  this --selftest           check the GPU code first (every line must say PASS)\n"
"  this --help               this text\n"
"\n"
"ENGINE - both match the ADDRESS; they differ only in the odds / time estimate\n"
"  --engine 1                plain estimate: every fixed character counts as 1 in 58\n"
"  --engine 2                Base58-aware estimate (default): the first character after the\n"
"                            leading 1 is only 2..Q (~1 in 23), later ones are 1 in 58\n"
"\n"
"PATTERN (case-sensitive; Base58 has no 0 O I l)\n"
"  --pattern P               ? = any one character, one * = any run of characters\n"
"                              1Kat*xyz       start 1Kat, end xyz\n"
"                              1Kat???xyz     ...no '*' and shorter than 34: everything up to the last ?\n"
"                                             is fixed from the start, what follows (xyz) is the END\n"
"                              1Ka???u???xyz  1Ka at 1-3, u at position 7, xyz = last three characters\n"
"                              1Kat           plain start (no ? in it)\n"
"                            exactly 34 characters = every character at its own position\n"
"  --prefix S                start of the address, e.g. 1Kat\n"
"  --suffix S                end of the address, e.g. xyz\n"
"  --pos N=C,N=C             fixed characters at 1-based positions 1-34, e.g. --pos 6=A,12=z\n"
"  (all four can be combined; conflicting characters are rejected; the resolved\n"
"   34-character pattern is printed before the search starts)\n"
"\n"
"PRIVATE-KEY RANGE (optional; without it a random 255-bit start is used and both k and n-k are tested)\n"
"  --start HEX               first private key, hex, minimum 1\n"
"  --end HEX                 last private key, hex, maximum n-1 =\n"
"                            FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140\n"
"                            (only --start given = scan from start to n-1; only --end = from 1)\n"
"  --random-start            begin at a random point inside the range, wrap around at the end\n"
"  The search walks the range one key per step and stops by itself with\n"
"  \"Range fully searched\" once every key has been tested.\n"
"\n"
"BIT RANGES (optional, instead of --start/--end)\n"
"  --bits 45,52,58           work through ranges of 45-bit keys (2^44..2^45-1), then 52-bit, then\n"
"                            58-bit, then back to the first. Each range remembers where it stopped.\n"
"  --jump-after N            move to the next range after N matches (default 1), or as soon as a\n"
"                            range is fully searched. The start-up table shows, per range, the chance\n"
"                            of at least one match (Engine 2 uses the Base58 probability, Engine 1 the\n"
"                            plain 1-in-58 estimate). This only schedules ranges: it does not make a\n"
"                            match more likely, since neighbouring keys give unrelated addresses.\n"
"\n"
"RUN CONTROL\n"
"  --count N                 stop after N matches. Default 0 = never stop on a match: keep going\n"
"                            until the range is exhausted (or forever without a range).\n"
"  --out FILE                every match is also appended to this file (default matches.txt)\n"
"  --checkpoint FILE         progress file, rewritten about once a second (default lastkey.ckpt)\n"
"  --resume                  continue from the checkpoint file (restores engine, pattern, range)\n"
"  --blocks N --threads N --iters N    set the launch size by hand (threads 32..128)\n"
"  --no-tune                 skip the start-up timing run that picks threads/blocks for your GPU\n"
"                            (it is also skipped when you set --blocks/--threads/--iters)\n"
"\n"
"LAST KEY SCANNED\n"
"  On Ctrl+C, kill, closing the terminal or a CUDA error the program prints 'Last key scanned'\n"
"  and 'Next key to scan'. If the machine loses power or the process is killed hard, read the\n"
"  same values from the checkpoint file. All keys up to 'last' have been tested.\n"
"\n"
"EXAMPLES\n"
"  this --selftest\n"
"  this --pattern \"1Kat???xyz\"\n"
"  this --engine 1 --prefix 1Kat --suffix xyz\n"
"  this --pattern \"1Kat*\" --start 1 --end ffffffffff\n"
"  this --pattern \"1Kat*\" --bits 45,52,58 --jump-after 1\n"
"  this --resume\n"
"\n"
"Keys from a small or known range can be guessed by anyone: do not hold real funds on them.\n");
}

int main(int argc, char** argv) {
  std::string prefix, suffix, pattern, posList, startHex, endHex, bitsStr; int jumpAfter = 1;
  int engine = 0, blocks = 0, threads = 128, iters = 64, wantHits = 0; bool st = false, randStart = false, resume = false, autoTune = true;
  std::string outFile = "matches.txt", ckptFile = "lastkey.ckpt";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if      (a == "--engine"  && i + 1 < argc) engine = atoi(argv[++i]);
    else if (a == "--prefix"  && i + 1 < argc) prefix = argv[++i];
    else if (a == "--suffix"  && i + 1 < argc) suffix = argv[++i];
    else if (a == "--pattern" && i + 1 < argc) pattern = argv[++i];
    else if (a == "--pos"     && i + 1 < argc) posList = argv[++i];
    else if (a == "--start"   && i + 1 < argc) startHex = argv[++i];
    else if (a == "--end"     && i + 1 < argc) endHex = argv[++i];
    else if (a == "--bits"    && i + 1 < argc) bitsStr = argv[++i];
    else if (a == "--jump-after" && i + 1 < argc) jumpAfter = std::max(1, atoi(argv[++i]));
    else if (a == "--random-start") randStart = true;
    else if (a == "--blocks"  && i + 1 < argc) { blocks = atoi(argv[++i]); autoTune = false; }
    else if (a == "--threads" && i + 1 < argc) { threads = atoi(argv[++i]); autoTune = false; }
    else if (a == "--iters"   && i + 1 < argc) { iters = atoi(argv[++i]); autoTune = false; }
    else if (a == "--no-tune") autoTune = false;
    else if (a == "--count"   && i + 1 < argc) wantHits = std::max(0, atoi(argv[++i]));
    else if (a == "--out"     && i + 1 < argc) outFile = argv[++i];
    else if (a == "--checkpoint" && i + 1 < argc) ckptFile = argv[++i];
    else if (a == "--resume") resume = true;
    else if (a == "--selftest") st = true;
    else if (a == "--help" || a == "-h" || a == "-?") { usage(); return 0; }
    else { fprintf(stderr, "unknown or incomplete argument: %s (try --help)\n", a.c_str()); return 1; }
  }
  if (!bitsStr.empty() && (!startHex.empty() || !endHex.empty())) {
    fprintf(stderr, "error: use either --bits or --start/--end, not both\n"); return 1;
  }
  bool resumed = false; u64 ck_next[4] = {0, 0, 0, 0}; double ck_scanned = 0;
  int ck_ridx = 0; std::vector<std::string> ck_rnext; std::vector<int> ck_rdone; std::vector<double> ck_rscan;
  if (resume) {
    std::map<std::string, std::string> kv;
    if (!read_kv(ckptFile, kv)) { fprintf(stderr, "error: cannot read checkpoint file '%s'\n", ckptFile.c_str()); return 1; }
    if (!kv.count("start") || !kv.count("end") || !kv.count("next") || !parse_hex256(kv["next"], ck_next)) {
      fprintf(stderr, "error: checkpoint file '%s' is incomplete\n", ckptFile.c_str()); return 1;
    }
    if (kv.count("done") && kv["done"] == "1") { printf("The checkpointed run already finished (range fully searched).\n"); return 0; }
    randStart = false;
    if (kv.count("bits")) {                                // bit-range schedule: restore every range's position
      bitsStr = kv["bits"]; if (kv.count("jump")) jumpAfter = std::max(1, atoi(kv["jump"].c_str()));
      if (kv.count("ridx")) ck_ridx = atoi(kv["ridx"].c_str());
      for (int i = 0; kv.count("rnext_" + std::to_string(i)); i++) {
        ck_rnext.push_back(kv["rnext_" + std::to_string(i)]);
        ck_rdone.push_back(kv.count("rdone_" + std::to_string(i)) ? atoi(kv["rdone_" + std::to_string(i)].c_str()) : 0);
        ck_rscan.push_back(kv.count("rscan_" + std::to_string(i)) ? atof(kv["rscan_" + std::to_string(i)].c_str()) : 0.0);
      }
    } else { startHex = kv["start"]; endHex = kv["end"]; }
    if (prefix.empty() && suffix.empty() && pattern.empty() && posList.empty() && kv.count("pattern")) {
      pattern = kv["pattern"]; if (engine == 0 && kv.count("engine")) engine = atoi(kv["engine"].c_str());
    }
    if (kv.count("scanned")) ck_scanned = atof(kv["scanned"].c_str());
    resumed = true;
    printf("Resuming from %s (last key scanned: %s)\n", ckptFile.c_str(), kv.count("last") ? kv["last"].c_str() : "none");
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  if (threads > 128) threads = 128;                       // __launch_bounds__(128)
  if (threads < 32) threads = 32;
  if (iters < 1) iters = 1;
  if (blocks == 0) blocks = prop.multiProcessorCount * 4;

  build_table<<<1, BATCH>>>(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  { std::vector<pt> htab(BATCH + 1);                      // copy the table into constant memory
    CK(cudaMemcpyFromSymbol(htab.data(), d_tab, sizeof(pt) * (BATCH + 1)));
    CK(cudaMemcpyToSymbol(c_tab, htab.data(), sizeof(pt) * (BATCH + 1))); }
  { u32 m33[6]; min33_words(m33); CK(cudaMemcpyToSymbol(c_min33, m33, sizeof m33)); }
  if (st) {
    const u64 M4 = pow58(4), mu4 = (~0ULL) / M4;
    selftest<<<1, 1>>>(M4, mu4); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    u64 acc, acc2; int is34; unsigned char dig[33];
    CK(cudaMemcpyFromSymbol(&acc, d_acc, sizeof acc)); CK(cudaMemcpyFromSymbol(dig, d_dig, sizeof dig));
    CK(cudaMemcpyFromSymbol(&acc2, d_acc2, sizeof acc2)); CK(cudaMemcpyFromSymbol(&is34, d_is34, sizeof is34));
    unsigned char hg[21] = {0x00,0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
    std::string addr = b58check(std::vector<unsigned char>(hg, hg + 21));
    printf("host address          : %s (%s)\n", addr.c_str(), addr == "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH" ? "PASS" : "FAIL");
    u64 want = b58val(addr.substr(addr.size() - 4));
    printf("GPU suffix arithmetic : %s\n", acc == want ? "PASS" : "FAIL");
    bool digOk = addr.size() == ALEN;
    for (int i = 1; i < ALEN && digOk; i++) digOk = (b58idx(addr[i]) == dig[ALEN - 1 - i]);
    printf("GPU Base58 digits     : %s\n", digOk ? "PASS" : "FAIL");
    printf("GPU fast suffix mod   : %s\n", acc2 == want ? "PASS" : "FAIL");
    printf("GPU 34-character check: %s\n", is34 == 1 ? "PASS" : "FAIL");
    signed char tp[ALEN]; memset(tp, -1, sizeof tp);       // fixed characters taken from the real address
    const int fixedPos[4] = {6, 12, 20, 30};
    for (int q = 0; q < 4; q++) tp[fixedPos[q]] = (signed char)b58idx(addr[fixedPos[q]]);
    int lg = last_group(tp);
    CK(cudaMemcpyToSymbol(c_pat, tp, sizeof tp)); CK(cudaMemcpyToSymbol(c_lastGroup, &lg, 4));
    selftest_digits<<<1, 1>>>(1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    tp[12] = (signed char)((tp[12] + 1) % 58);              // one wrong character must be rejected
    CK(cudaMemcpyToSymbol(c_pat, tp, sizeof tp));
    selftest_digits<<<1, 1>>>(0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    return 0;
  }

  auto ask = [](const char* msg, std::string& out) {
    char buf[256]; printf("%s", msg); fflush(stdout);
    if (fgets(buf, sizeof buf, stdin)) {
      out = buf;
      while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
    }
  };
  if (prefix.empty() && suffix.empty() && pattern.empty() && posList.empty()) {      // interactive mode
    if (engine == 0) {
      std::string e;
      printf("Both options match the address; they differ only in the odds / time estimate:\n"
             "  1 = plain estimate (every fixed character counted as 1 in 58)\n"
             "  2 = Base58-aware estimate (first character after the 1 is only 2..Q)\n");
      ask("Choice [2]: ", e);
      engine = (!e.empty() && e[0] == '1') ? 1 : 2;
    }
    ask("Address pattern, e.g. 1Kat???xyz  (? = any character, * = any gap; Enter to skip): ", pattern);
    if (pattern.empty()) {
      ask("Address START (begins with 1, e.g. 1Kat; Enter to skip): ", prefix);
      ask("Address END   (e.g. xyz; Enter to skip): ", suffix);
      ask("Fixed characters by position 1-34 (e.g. 6=A,12=z; Enter to skip): ", posList);
    }
    if (startHex.empty() && endHex.empty()) {
      ask("Private key range START, hex (Enter = no range / random key space): ", startHex);
      ask("Private key range END,   hex (Enter = top of key space): ", endHex);
      if (startHex.empty() && endHex.empty())
        ask("Or bit ranges to work through in turn, e.g. 45,52,58 (Enter = none): ", bitsStr);
      if (!startHex.empty() || !endHex.empty() || !bitsStr.empty()) {
        std::string yn; ask("Begin at a random point inside the range? (y/N): ", yn);
        randStart = !yn.empty() && (yn[0] == 'y' || yn[0] == 'Y');
      }
    }
    std::string cn; ask("Stop after N matches (Enter = keep running until the range is exhausted): ", cn);
    if (!cn.empty()) wantHits = std::max(0, atoi(cn.c_str()));
  }
  if (engine == 0) engine = 2;
  if (engine != 1 && engine != 2) { fprintf(stderr, "error: --engine must be 1 or 2\n"); return 1; }

  // ---- what to look for
  std::string err, shown, note;
  Pattern pat; memset(&pat, 0, sizeof pat);
  int hasPrefix = 0, hasSuffix = 0, needFull = 0; u64 lo64 = 0, hi64 = ~0ULL, sufMod = 1, sufVal = 0;
  signed char patIdx[ALEN]; memset(patIdx, -1, sizeof patIdx);
  signed char devPat[ALEN]; memset(devPat, -1, sizeof devPat);   // digits the full decode still has to check
  double expect = 1;
  {
    if (!build_pattern(prefix, suffix, pattern, posList, pat, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    PatInfo info = analyze(pat);
    if (!info.hasPrefix && !info.hasSuffix && !info.needFull) {
      fprintf(stderr, "error: nothing to search for. Give at least one character after the leading '1'.\n"); return 1;
    }
    std::string digits; for (int i = 1; i <= info.m1; i++) digits.push_back(pat.c[i]);
    double Pwin = 1; u64 l = 0, h = ~0ULL;
    if (!prefix_window(digits, l, h, &Pwin)) {
      fprintf(stderr, "error: that start cannot occur in a 34-character address. After the leading 1 the next character can only be 2 to Q.\n"); return 1;
    }
    if (info.hasPrefix) { lo64 = l; hi64 = h; }
    hasPrefix = info.hasPrefix; hasSuffix = info.hasSuffix; needFull = info.needFull;
    std::string suf; for (int i = ALEN - info.k; i < ALEN; i++) suf.push_back(pat.c[i]);
    sufMod = info.hasSuffix ? pow58(info.k) : 1; sufVal = info.hasSuffix ? b58val(suf) : 0;
    for (int i = 0; i < ALEN; i++) patIdx[i] = pat.c[i] ? (signed char)b58idx(pat.c[i]) : (signed char)-1;
    memcpy(devPat, patIdx, sizeof devPat);                // the prefix window and the suffix test already cover their digits
    if (info.hasPrefix) for (int i = 1; i <= info.m1; i++) devPat[i] = -1;
    if (info.hasSuffix) for (int i = ALEN - info.k; i < ALEN; i++) devPat[i] = -1;
    if (engine == 1) {                                    // plain estimate
      expect = pow(58.0, info.fixed);
      note = " (plain estimate: every fixed character counted as 1 in 58)";
    } else {                                              // Base58-aware estimate
      expect = pow(58.0, info.fixed - info.m1) / Pwin;    // exact window probability for the leading characters
      note = " (Base58-aware: first character after the 1 is only 2..Q)";
    }
    for (int i = 0; i < ALEN; i++) shown.push_back(pat.c[i] ? pat.c[i] : '?');
    if (!info.hasPrefix && info.hasSuffix && !info.needFull) note += " | suffix-only: slower per key";
    if (info.needFull) note += " | fixed middle positions: every key is decoded, slower per key";
  }

  // ---- private-key range
  std::vector<int> bitsList;                              // --bits 45,52,58 = keys of 45 bits, then 52 bits, ...
  {
    size_t i = 0;
    while (i < bitsStr.size()) {
      while (i < bitsStr.size() && (bitsStr[i] == ',' || bitsStr[i] == ' ' || bitsStr[i] == ';')) i++;
      if (i >= bitsStr.size()) break;
      size_t j = i; while (j < bitsStr.size() && bitsStr[j] != ',' && bitsStr[j] != ' ' && bitsStr[j] != ';') j++;
      int b = atoi(bitsStr.substr(i, j - i).c_str()); i = j;
      if (b < 1 || b > 255) { fprintf(stderr, "error: --bits values must be between 1 and 255\n"); return 1; }
      bitsList.push_back(b);
    }
    if (!bitsStr.empty() && bitsList.empty()) { fprintf(stderr, "error: bad --bits list (use e.g. 45,52,58)\n"); return 1; }
  }
  const bool bitsMode = !bitsList.empty();
  const int nb = (int)bitsList.size();
  bool rangeMode = bitsMode || !startHex.empty() || !endHex.empty();
  u64 rs[4] = {1, 0, 0, 0}, re[4], size[4]; memcpy(re, NMINUS1, sizeof re);
  double sizeD = 0;
  int ridx = 0;
  // b-bit keys: 2^(b-1) .. 2^b - 1
  auto setBitRange = [&](int b) {
    memset(rs, 0, sizeof rs); memset(re, 0, sizeof re);
    rs[(b - 1) / 64] = 1ULL << ((b - 1) % 64);
    for (int w = 0; w < 4; w++) re[w] = b >= 64 * (w + 1) ? ~0ULL : (b > 64 * w ? (1ULL << (b - 64 * w)) - 1 : 0ULL);
  };
  if (bitsMode) {
    if (resumed) ridx = (ck_ridx >= 0 && ck_ridx < nb) ? ck_ridx : 0;
    setBitRange(bitsList[ridx]);
  }
  if (rangeMode) {
    if (!startHex.empty() && !parse_hex256(startHex, rs)) { fprintf(stderr, "error: bad --start hex\n"); return 1; }
    if (!endHex.empty() && !parse_hex256(endHex, re))     { fprintf(stderr, "error: bad --end hex\n"); return 1; }
    if (is_zero256(rs)) { fprintf(stderr, "error: range start must be at least 1\n"); return 1; }
    if (cmp256(re, NMINUS1) > 0) { fprintf(stderr, "error: range end must be below the curve order n (FFFF...FFFE BAAEDCE6...D0364140)\n"); return 1; }
    if (cmp256(rs, re) > 0) { fprintf(stderr, "error: range start is above range end\n"); return 1; }
    sub256(size, re, rs); add256_u64(size, 1);
    sizeD = to_double256(size);
    printf("Key range: "); print_be(rs); printf(" .. "); print_be(re); printf("  (~%.3g keys)\n", sizeD);
    if (bitsMode) {
      printf("Bit-range schedule (%s estimate): moves to the next range after %d match%s, or when a range is exhausted\n",
             engine == 1 ? "plain" : "Base58-aware", jumpAfter, jumpAfter == 1 ? "" : "es");
      for (int i = 0; i < nb; i++) {
        double sz = ldexp(1.0, bitsList[i] - 1);
        printf("  #%d  %3d-bit keys: ~%.3g keys, chance of at least one match in the range %.4g%%%s\n", i + 1, bitsList[i], sz,
               -expm1(-sz / expect) * 100.0, i == ridx ? "   <- start here" : "");
      }
    }
    printf("WARNING: keys from a small or known range can be guessed by anyone. Do not use them to hold real funds.\n");
  }

  // ---- constants for the kernels
  int rangeFlag = rangeMode;
  u64 sufMu = sufMod > 1 ? (~0ULL) / sufMod : 0;
  int lastGroup = last_group(devPat);
  CK(cudaMemcpyToSymbol(c_lo64, &lo64, 8)); CK(cudaMemcpyToSymbol(c_hi64, &hi64, 8));
  CK(cudaMemcpyToSymbol(c_sufMod, &sufMod, 8)); CK(cudaMemcpyToSymbol(c_sufVal, &sufVal, 8)); CK(cudaMemcpyToSymbol(c_sufMu, &sufMu, 8));
  CK(cudaMemcpyToSymbol(c_hasPrefix, &hasPrefix, 4)); CK(cudaMemcpyToSymbol(c_hasSuffix, &hasSuffix, 4));
  CK(cudaMemcpyToSymbol(c_needFull, &needFull, 4)); CK(cudaMemcpyToSymbol(c_range, &rangeFlag, 4));
  CK(cudaMemcpyToSymbol(c_pat, devPat, sizeof devPat)); CK(cudaMemcpyToSymbol(c_lastGroup, &lastGroup, 4));

  // ---- launch geometry
  const double sizeGeom = bitsMode ? ldexp(1.0, *std::max_element(bitsList.begin(), bitsList.end()) - 1) : sizeD;
  if (autoTune && !(rangeMode && sizeGeom < 4e9)) {       // short timing run: picks threads per block and blocks per SM
    const int sm = prop.multiProcessorCount, thrC[2] = {64, 128}, mulC[3] = {2, 4, 8}, tIters = 16;
    const size_t maxT = (size_t)sm * 8 * 128;
    fe *tx, *ty; Hit* th; u32* tc;
    CK(cudaMalloc(&tx, maxT * sizeof(fe))); CK(cudaMalloc(&ty, maxT * sizeof(fe)));
    CK(cudaMalloc(&th, MAXHITS * sizeof(Hit))); CK(cudaMalloc(&tc, sizeof(u32)));
    fe tb; memset(&tb, 0, sizeof tb); tb.v[0] = 0x1234567ULL;            // any valid key: results are thrown away
    const u64 tStride = (u64)BATCH * tIters;
    double bestRate = 0; int bestB = sm * 4, bestT = 128;
    printf("Tuning launch size (a few seconds, add --no-tune to skip)...\n");
    for (int ti = 0; ti < 2; ti++) for (int mi = 0; mi < 3; mi++) {
      int bl = sm * mulC[mi], th_ = thrC[ti]; double dt = 0;
      for (int rep = 0; rep < 2; rep++) {                                // first run warms up, second is timed
        CK(cudaMemset(tc, 0, sizeof(u32)));
        auto t1 = std::chrono::steady_clock::now();
        init_points<<<bl, th_>>>(tb, tStride, tx, ty);
        search<<<bl, th_>>>(tx, ty, tIters, tStride, th, tc);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
      }
      double rate = (double)tStride * bl * th_ / dt;
      printf("  threads %3d x blocks %4d : %s\n", th_, bl, fmt_rate(rate).c_str());
      if (rate > bestRate * 1.02) { bestRate = rate; bestB = bl; bestT = th_; }   // ties keep the earlier (smaller) choice
    }
    blocks = bestB; threads = bestT;
    CK(cudaFree(tx)); CK(cudaFree(ty)); CK(cudaFree(th)); CK(cudaFree(tc));
    printf("Tuned   : threads %d, blocks %d\n", threads, blocks);
  }
  if (rangeMode && sizeGeom < (double)BATCH * blocks * threads * iters) {   // shrink for tiny ranges
    blocks = (int)std::max(1.0, std::min((double)blocks, std::ceil(sizeGeom / ((double)BATCH * threads))));
    iters  = (int)std::max(1.0, std::min((double)iters,  std::ceil(sizeGeom / ((double)BATCH * blocks * threads))));
  }
  size_t nthreads = (size_t)blocks * threads;
  u64 stride = (u64)BATCH * iters;
  u64 perLaunch = stride * nthreads;

  fe *d_px, *d_py; Hit* d_hits; u32* d_cnt;
  CK(cudaMalloc(&d_px, nthreads * sizeof(fe))); CK(cudaMalloc(&d_py, nthreads * sizeof(fe)));
  CK(cudaMalloc(&d_hits, MAXHITS * sizeof(Hit))); CK(cudaMalloc(&d_cnt, sizeof(u32)));

  fe base;
  if (rangeMode) {
    u64 r0[4] = {0, 0, 0, 0};
    if (randStart) { u64 diff[4]; sub256(diff, re, rs); rand256_upto(diff, r0); }
    add256(base.v, rs, r0);
    if (resumed) {
      if (cmp256(ck_next, rs) >= 0 && cmp256(ck_next, re) <= 0) memcpy(base.v, ck_next, sizeof base.v);
      else fprintf(stderr, "warning: checkpoint position is outside the range, starting at the range start\n");
    }
  } else {                                                // random 255-bit start from the OS entropy source
    std::random_device rd;
    for (int i = 0; i < 4; i++) base.v[i] = ((u64)rd() << 32) | rd();
    base.v[3] &= 0x7FFFFFFFFFFFFFFFULL;
  }

  // per-range state for --bits (position, finished flag, keys scanned)
  std::vector<std::array<u64, 4>> rNext(nb > 0 ? nb : 1); std::vector<int> rDone(nb > 0 ? nb : 1, 0); std::vector<double> rScan(nb > 0 ? nb : 1, 0.0);
  std::vector<char> rInit(nb > 0 ? nb : 1, 0);
  if (bitsMode && resumed) {
    for (int i = 0; i < nb && i < (int)ck_rnext.size(); i++) {
      u64 t[4]; if (!parse_hex256(ck_rnext[i], t)) continue;
      memcpy(rNext[i].data(), t, sizeof t); rDone[i] = ck_rdone[i]; rScan[i] = ck_rscan[i]; rInit[i] = 1;
    }
  }
  if (bitsMode) { memcpy(rNext[ridx].data(), base.v, sizeof base.v); rInit[ridx] = 1; }

  printf("Engine  : %d (address match, %s estimate)\n", engine, engine == 1 ? "plain" : "Base58-aware");
  printf("Pattern : %s\n", shown.c_str());
  printf("GPU     : %s | threads %zu | %.3g keys per launch\n", prop.name, nthreads, (double)perLaunch * (rangeMode ? 1.0 : 2.0));
  printf("Expect  : ~%.3g keys per match%s\n", expect, note.c_str());
  printf("Odds    : p = %.3e per key (1 in %.3g)", 1.0 / expect, expect);
  if (rangeMode) printf(" | chance of at least one match in the whole range: %.4g%%", -expm1(-sizeD / expect) * 100.0);
  printf("\n");
  if (rangeMode) printf("Start at: %s\n", hex_short(base.v).c_str());

  auto matches = [&](const std::string& addr, const std::string&) { return pattern_match(addr, pat); };

  int found = 0; double total = 0, scanned = resumed ? ck_scanned : 0; bool exhausted = false;
  const double keysPerLaunch = (double)perLaunch * (rangeMode ? 1.0 : 2.0);   // non-range: x tests k and n-k
  int foundInRange = 0;
  auto syncSt = [&]() {                                   // copy the per-range schedule into the checkpoint state
    if (!bitsMode) return;
    g_st.bits = bitsStr; g_st.jumpAfter = jumpAfter; g_st.ridx = ridx;
    g_st.rnext.assign(nb, std::string()); g_st.rdone.assign(nb, 0); g_st.rscan.assign(nb, 0.0);
    for (int i = 0; i < nb; i++) { g_st.rnext[i] = hex256(rNext[i].data()); g_st.rdone[i] = rDone[i]; g_st.rscan[i] = rScan[i]; }
  };
  g_st.active = true; g_st.range = rangeMode; g_st.engine = engine; g_st.pattern = shown; g_st.ckpt = ckptFile;
  memcpy(g_st.rs, rangeMode ? rs : base.v, sizeof g_st.rs); memcpy(g_st.re, re, sizeof g_st.re);
  memcpy(g_st.next, base.v, sizeof g_st.next); g_st.scanned = scanned;
  syncSt();
  atexit(report_last_key);
  signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
#ifdef SIGBREAK
  signal(SIGBREAK, on_signal);                            // Windows: Ctrl+Break / console window closed
#endif
#ifdef SIGHUP
  signal(SIGHUP, on_signal);                              // terminal closed
#endif
  write_checkpoint();
  printf("Running %s. Ctrl+C stops it and prints the last key scanned (also kept in %s).\n",
         wantHits ? "until the requested matches are found" : (rangeMode ? "until the range is exhausted" : "until stopped"),
         ckptFile.c_str());
  auto t0 = std::chrono::steady_clock::now(), lastSave = t0;
  while ((wantHits == 0 || found < wantHits) && !g_stop) {
    auto tl = std::chrono::steady_clock::now();
    CK(cudaMemset(d_cnt, 0, sizeof(u32)));
    init_points<<<blocks, threads>>>(base, stride, d_px, d_py);
    search<<<blocks, threads>>>(d_px, d_py, iters, stride, d_hits, d_cnt);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    auto tn = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(tn - tl).count();
    u32 cnt; CK(cudaMemcpy(&cnt, d_cnt, sizeof cnt, cudaMemcpyDeviceToHost));
    if (cnt > MAXHITS) cnt = MAXHITS;
    if (cnt) {
      Hit hits[MAXHITS]; CK(cudaMemcpy(hits, d_hits, cnt * sizeof(Hit), cudaMemcpyDeviceToHost));
      for (u32 i = 0; i < cnt && (wantHits == 0 || found < wantHits); i++) {
        std::vector<unsigned char> payload(21, 0);        // version 0x00 || hash160
        char hx[41];
        for (int w = 0; w < 5; w++) for (int b = 0; b < 4; b++) {
          unsigned char by = (hits[i].h[w] >> (8 * b)) & 0xff;
          payload[1 + 4 * w + b] = by; snprintf(hx + 8 * w + 2 * b, 3, "%02x", by);
        }
        std::string addr = b58check(payload);
        if (!matches(addr, std::string(hx))) continue;    // host re-check
        bool neg = (hits[i].off & NEGFLAG) != 0; u64 off = hits[i].off & ~NEGFLAG;
        u64 k[4]; memcpy(k, base.v, sizeof k); add256_u64(k, off);
        if (neg) sub256(k, ORDER_N, k);                   // key is n - k
        if (rangeMode && (cmp256(k, rs) < 0 || cmp256(k, re) > 0)) continue;   // overshoot past the range end
        std::vector<unsigned char> wif(34); wif[0] = 0x80; // 0x80 || key(32, big-endian) || 0x01 (compressed)
        for (int j = 0; j < 32; j++) wif[1 + j] = (unsigned char)(k[3 - j / 8] >> (56 - 8 * (j % 8)));
        wif[33] = 0x01;
        std::string wifs = b58check(wif);
        printf("\nMATCH  address = %s\n       hash160 = %s\n       privkey = ", addr.c_str(), hx); print_be(k);
        printf("\n       WIF     = %s\n", wifs.c_str());
        fflush(stdout);
        if (FILE* of = fopen(outFile.c_str(), "a")) {      // keep every match, the run may go on for days
          fprintf(of, "address=%s hash160=%s privkey=%s WIF=%s\n", addr.c_str(), hx, hex256(k).c_str(), wifs.c_str());
          fclose(of);
        }
        found++; foundInRange++;
      }
    }
    double eff = keysPerLaunch;                           // keys of this launch that lie inside the range
    if (rangeMode) { u64 rem[4]; sub256(rem, re, base.v); double r = to_double256(rem) + 1.0; if (r < eff) eff = r; }
    total += eff; scanned += rangeMode ? eff : (double)perLaunch;
    u64 lastk[4]; memcpy(lastk, base.v, sizeof lastk);    // every key up to here has been tested
    u64 cy = add256_u64(lastk, perLaunch - 1);
    if (rangeMode && (cy || cmp256(lastk, re) > 0)) memcpy(lastk, re, sizeof lastk);
    double secs = std::chrono::duration<double>(tn - t0).count();
    double avg = total / secs, inst = eff / dt;
    printf("\r%s (avg %s) | %.3g keys | %s | ~%s/match | found %d", fmt_rate(inst).c_str(), fmt_rate(avg).c_str(), total,
           fmt_dur(secs).c_str(), fmt_dur(expect / avg).c_str(), found);
    if (bitsMode) printf(" | %d-bit", bitsList[ridx]);
    if (rangeMode) printf(" | range %.2f%%", std::min(100.0, 100.0 * scanned / sizeD));
    printf(" | last %s      ", hex_short(lastk).c_str()); fflush(stdout);
    u64 carry = add256_u64(base.v, perLaunch);            // advance; wrap inside the range
    if (rangeMode && (carry || cmp256(base.v, re) > 0)) memcpy(base.v, rs, sizeof base.v);
    memcpy(g_st.last, lastk, sizeof lastk); g_st.haveLast = true; memcpy(g_st.next, base.v, sizeof base.v);
    g_st.scanned = scanned; g_st.found = found;
    if (bitsMode) {
      bool fin = scanned >= sizeD;
      if (fin) rDone[ridx] = 1;
      bool allDone = true; for (int i = 0; i < nb; i++) if (!rDone[i]) allDone = false;
      if (allDone) { exhausted = true; g_st.done = true; }
      else if (fin || foundInRange >= jumpAfter) {
        memcpy(rNext[ridx].data(), base.v, sizeof base.v); rScan[ridx] = scanned;   // remember where this range stopped
        int nx = ridx; do { nx = (nx + 1) % nb; } while (rDone[nx]);
        if (nx != ridx) {
          ridx = nx; setBitRange(bitsList[ridx]);
          sub256(size, re, rs); add256_u64(size, 1); sizeD = to_double256(size);
          if (rInit[ridx]) memcpy(base.v, rNext[ridx].data(), sizeof base.v);
          else {
            u64 r0[4] = {0, 0, 0, 0};
            if (randStart) { u64 diff[4]; sub256(diff, re, rs); rand256_upto(diff, r0); }
            add256(base.v, rs, r0); rInit[ridx] = 1;
          }
          scanned = rScan[ridx];
          memcpy(g_st.rs, rs, sizeof g_st.rs); memcpy(g_st.re, re, sizeof g_st.re);
          memcpy(g_st.next, base.v, sizeof g_st.next); g_st.scanned = scanned;
          printf("\nJump -> %d-bit range (#%d of %d), continuing at %s\n", bitsList[ridx], ridx + 1, nb, hex_short(base.v).c_str());
        }
        foundInRange = 0;
      }
      syncSt();
    } else if (rangeMode && scanned >= sizeD) { exhausted = true; g_st.done = true; }
    if (exhausted || std::chrono::duration<double>(tn - lastSave).count() >= 1.0) { write_checkpoint(); lastSave = tn; }
    if (exhausted) break;
  }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("\n%s%s%.0f keys in %s, average %.0f keys/s (%s), %d match(es)\n", exhausted ? "Range fully searched. " : "",
         g_stop ? "Interrupted. " : "", total, fmt_dur(secs).c_str(), total / secs, fmt_rate(total / secs).c_str(), found);
  return 0;                                               // report_last_key() prints the last key on exit
}
#endif  // VANITY_HOST_TEST
