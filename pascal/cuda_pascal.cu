// Original version written by Schleicher (KlausT @github)
// Redistribution and use in source and binary forms, with or without modification, are permitted

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include "miner.h"
#include "cuda_helper.h"

void pascal_cpu_init(int thr_id);
void pascal_cpu_hash(int thr_id, uint32_t threads, uint32_t *data, uint32_t datasize, uint32_t *ms, uint32_t *const h_result);
void pascal_midstate(const uint32_t *data, uint32_t *midstate);

static THREAD uint32_t *d_result;

#define TPB 512
#define NONCES_PER_THREAD 32

__global__ __launch_bounds__(TPB, 2)
void pascal_gpu_hash(const uint32_t threads, uint32_t *const result, const uint32_t *data, const uint32_t datasize, uint32_t ms0, const uint32_t ms1, const uint32_t ms2, const uint32_t ms3, const uint32_t ms4, const uint32_t ms5, const uint32_t ms6, const uint32_t ms7)
{
	const uint32_t threadindex = (blockDim.x * blockIdx.x + threadIdx.x);
	if (threadindex < threads)
	{
		uint32_t s0, s1, t1, a, b, c, d, e, f, g, h;
		uint32_t w[64];
		const uint32_t startnonce = data[datasize / 4 - 1];
		const uint32_t numberofthreads = blockDim.x*gridDim.x;
		const uint32_t maxnonce = startnonce + threadindex + numberofthreads*NONCES_PER_THREAD - 1;

		for (uint32_t nonce = startnonce + threadindex; nonce <= maxnonce; nonce += numberofthreads)
		{
#pragma unroll
			for(int i = 0; i <= 15; i++)
				w[i] = data[i];
			w[datasize / 4 - 1] = nonce;
#pragma unroll
			for(int i = 16; i <= 63; i++)
			{
				s0 = ROTR32(w[i - 15], 7) ^ ROTR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
				s1 = ROTR32(w[i - 2], 17) ^ ROTR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
				w[i] = w[i - 16] + s0 + w[i - 7] + s1;
			}

			a = ms0;
			b = ms1;
			c = ms2;
			d = ms3;
			e = ms4;
			f = ms5;
			g = ms6;
			h = ms7;

			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x428a2f98U + w[0];
			d = d + t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x71374491U + w[1];
			c = c + t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xb5c0fbcfU + w[2];
			b = b + t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xe9b5dba5U + w[3];
			a = a + t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x3956c25bU + w[4];
			h = h + t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x59f111f1U + w[5];
			g = g + t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x923f82a4U + w[6];
			f = f + t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xab1c5ed5U + w[7];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xd807aa98U + w[8];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x12835b01U + w[9];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x243185beU + w[10];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x550c7dc3U + w[11];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x72be5d74U + w[12];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x80deb1feU + w[13];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x9bdc06a7U + w[14];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xc19bf3f4U + w[15];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xe49b69c1U + w[16];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xefbe4786U + w[17];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x0fc19dc6U + w[18];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x240ca1ccU + w[19];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x2de92c6fU + w[20];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4a7484aaU + w[21];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5cb0a9dcU + w[22];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x76f988daU + w[23];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x983e5152U + w[24];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xa831c66dU + w[25];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xb00327c8U + w[26];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xbf597fc7U + w[27];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xc6e00bf3U + w[28];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xd5a79147U + w[29];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x06ca6351U + w[30];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x14292967U + w[31];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x27b70a85U + w[32];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x2e1b2138U + w[33];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x4d2c6dfcU + w[34];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x53380d13U + w[35];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x650a7354U + w[36];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x766a0abbU + w[37];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x81c2c92eU + w[38];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x92722c85U + w[39];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xa2bfe8a1U + w[40];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xa81a664bU + w[41];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xc24b8b70U + w[42];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xc76c51a3U + w[43];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xd192e819U + w[44];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xd6990624U + w[45];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0xf40e3585U + w[46];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x106aa070U + w[47];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x19a4c116U + w[48];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x1e376c08U + w[49];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x2748774cU + w[50];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x34b0bcb5U + w[51];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x391c0cb3U + w[52];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4ed8aa4aU + w[53];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5b9cca4fU + w[54];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x682e6ff3U + w[55];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x748f82eeU + w[56];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x78a5636fU + w[57];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x84c87814U + w[58];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x8cc70208U + w[59];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x90befffaU + w[60];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xa4506cebU + w[61];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0xbef9a3f7U + w[62];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xc67178f2U + w[63];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			w[0] = a + ms0; w[1] = b + ms1; w[2] = c + ms2; w[3] = d + ms3;
			w[4] = e + ms4; w[5] = f + ms5; w[6] = g + ms6; w[7] = h + ms7;
			// hash the hash ***************************************************************
			w[16] = w[0] + (ROTR32(w[1], 7) ^ ROTR32(w[1], 18) ^ (w[1] >> 3));
			w[17] = w[1] + (ROTR32(w[2], 7) ^ ROTR32(w[2], 18) ^ (w[2] >> 3)) + (ROTR32(0x100, 17) ^ ROTR32(0x100, 19) ^ (0x100 >> 10));
			w[18] = w[2] + (ROTR32(w[3], 7) ^ ROTR32(w[3], 18) ^ (w[3] >> 3)) + (ROTR32(w[16], 17) ^ ROTR32(w[16], 19) ^ (w[16] >> 10));
			w[19] = w[3] + (ROTR32(w[4], 7) ^ ROTR32(w[4], 18) ^ (w[4] >> 3)) + (ROTR32(w[17], 17) ^ ROTR32(w[17], 19) ^ (w[17] >> 10));
			w[20] = w[4] + (ROTR32(w[5], 7) ^ ROTR32(w[5], 18) ^ (w[5] >> 3)) + (ROTR32(w[18], 17) ^ ROTR32(w[18], 19) ^ (w[18] >> 10));
			w[21] = w[5] + (ROTR32(w[6], 7) ^ ROTR32(w[6], 18) ^ (w[6] >> 3)) + (ROTR32(w[19], 17) ^ ROTR32(w[19], 19) ^ (w[19] >> 10));
			w[22] = w[6] + 0x100 + (ROTR32(w[7], 7) ^ ROTR32(w[7], 18) ^ (w[7] >> 3)) + (ROTR32(w[20], 17) ^ ROTR32(w[20], 19) ^ (w[20] >> 10));
			w[23] = w[7] + w[16] + 0x11002000U + (ROTR32(w[21], 17) ^ ROTR32(w[21], 19) ^ (w[21] >> 10));
			w[24] = 0x80000000U + w[17] + (ROTR32(w[22], 17) ^ ROTR32(w[22], 19) ^ (w[22] >> 10));
			w[25] = w[18] + (ROTR32(w[23], 17) ^ ROTR32(w[23], 19) ^ (w[23] >> 10));
			w[26] = w[19] + (ROTR32(w[24], 17) ^ ROTR32(w[24], 19) ^ (w[24] >> 10));
			w[27] = w[20] + (ROTR32(w[25], 17) ^ ROTR32(w[25], 19) ^ (w[25] >> 10));
			w[28] = w[21] + (ROTR32(w[26], 17) ^ ROTR32(w[26], 19) ^ (w[26] >> 10));
			w[29] = w[22] + (ROTR32(w[27], 17) ^ ROTR32(w[27], 19) ^ (w[27] >> 10));
			w[30] = w[23] + (ROTR32(0x100, 7) ^ ROTR32(0x100, 18) ^ (0x100 >> 3)) + (ROTR32(w[28], 17) ^ ROTR32(w[28], 19) ^ (w[28] >> 10));
			w[31] = 0x100 + w[24] + (ROTR32(w[16], 7) ^ ROTR32(w[16], 18) ^ (w[16] >> 3)) + (ROTR32(w[29], 17) ^ ROTR32(w[29], 19) ^ (w[29] >> 10));
#pragma unroll
			for (int i = 32; i < 59; i++)
				w[i] = w[i - 16] + w[i - 7] + (ROTR32(w[i - 15], 7) ^ ROTR32(w[i - 15], 18) ^ (w[i - 15] >> 3)) + (ROTR32(w[i - 2], 17) ^ ROTR32(w[i - 2], 19) ^ (w[i - 2] >> 10));

			d = 0x98c7e2a2U + w[0];
			h = 0xfc08884dU + w[0];
			//
			t1 = (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (0x9b05688cU ^ (d & 0xca0b3af3)) + 0x90bb1e3cU + w[1];
			c = 0x3c6ef372U + t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + (0x2A01A605 | (h & 0xfb6feee7));
			//
			t1 = (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (0x510e527fU ^ (c & (d ^ 0x510e527fU))) + 0x50C6645BU + w[2];
			b = 0xbb67ae85U + t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((0x6a09e667U & h) | (g & (0x6a09e667U | h)));
			//
			t1 = (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x3AC42E24U + w[3];
			a = 0x6a09e667U + t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x3956c25bU + w[4];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x59f111f1U + w[5];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x923f82a4U + w[6];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xab1c5ed5U + w[7];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x5807aa98U;
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x12835b01U;
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x243185beU;
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x550c7dc3U;
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x72be5d74U;
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x80deb1feU;
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x9bdc06a7U;
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xc19bf274U;
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xe49b69c1U + w[16];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xefbe4786U + w[17];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x0fc19dc6U + w[18];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x240ca1ccU + w[19];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x2de92c6fU + w[20];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4a7484aaU + w[21];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5cb0a9dcU + w[22];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x76f988daU + w[23];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x983e5152U + w[24];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xa831c66dU + w[25];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xb00327c8U + w[26];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xbf597fc7U + w[27];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xc6e00bf3U + w[28];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xd5a79147U + w[29];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x06ca6351U + w[30];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x14292967U + w[31];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x27b70a85U + w[32];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x2e1b2138U + w[33];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x4d2c6dfcU + w[34];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x53380d13U + w[35];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x650a7354U + w[36];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x766a0abbU + w[37];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x81c2c92eU + w[38];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x92722c85U + w[39];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xa2bfe8a1U + w[40];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xa81a664bU + w[41];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xc24b8b70U + w[42];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xc76c51a3U + w[43];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xd192e819U + w[44];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xd6990624U + w[45];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0xf40e3585U + w[46];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x106aa070U + w[47];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x19a4c116U + w[48];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			t1 = g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x1e376c08U + w[49];
			c += t1;
			g = t1 + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((b & a) | (h & (b | a)));
			//
			t1 = f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x2748774cU + w[50];
			b += t1;
			f = t1 + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((a & h) | (g & (a | h)));
			//
			t1 = e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x34b0bcb5U + w[51];
			a += t1;
			e = t1 + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((h & g) | (f & (h | g)));
			//
			t1 = d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x391c0cb3U + w[52];
			h += t1;
			d = t1 + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((g & f) | (e & (g | f)));
			//
			t1 = c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4ed8aa4aU + w[53];
			g += t1;
			c = t1 + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((f & e) | (d & (f | e)));
			//
			t1 = b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5b9cca4fU + w[54];
			f += t1;
			b = t1 + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((e & d) | (c & (e | d)));
			//
			t1 = a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x682e6ff3U + w[55];
			e += t1;
			a = t1 + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((d & c) | (b & (d | c)));
			//
			t1 = h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x748f82eeU + w[56];
			d += t1;
			h = t1 + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((c & b) | (a & (c | b)));
			//
			c += g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x78a5636fU + w[57];
			//
			b += f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x84c87814U + w[58];
			//
			a += e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x8cc70208U + w[43] + w[52] + (ROTR32(w[44], 7) ^ ROTR32(w[44], 18) ^ (w[44] >> 3)) + (ROTR32(w[57], 17) ^ ROTR32(w[57], 19) ^ (w[57] >> 10));
			//
			h += d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x90befffaU + w[44] + w[53] + (ROTR32(w[45], 7) ^ ROTR32(w[45], 18) ^ (w[45] >> 3)) + (ROTR32(w[58], 17) ^ ROTR32(w[58], 19) ^ (w[58] >> 10));
			//
			if (h == 0xa41f32e7)
			{
				uint32_t tmp = atomicCAS(result, 0, nonce);
				if (tmp != 0)
					result[1] = nonce;
			}
		} // nonce loop
	} // if thread<threads
}


__host__
void pascal_cpu_hash(int thr_id, uint32_t threads, uint32_t *data, uint32_t datasize, uint32_t *ms, uint32_t *h_result)
{

	uint32_t grid = (threads + TPB*NONCES_PER_THREAD - 1) / TPB / NONCES_PER_THREAD;

	CUDA_SAFE_CALL(cudaMemset(d_result, 0, 8));

	pascal_gpu_hash << <grid, TPB>>> (threads, d_result, data, datasize, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6], ms[7]);

	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

__host__
void pascal_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_result, 2 * sizeof(uint32_t)));
}

