#include "miner.h"
#include "cuda_helper.h"

void pascal_cpu_init(int thr_id);
void pascal_cpu_hash(int thr_id, uint32_t threads, uint32_t startnonce, uint32_t nonceoffset, uint32_t *ms, uint32_t *const result);
void pascal_midstate(const uint32_t *data, uint32_t *midstate);
void copydata(const uint32_t *data);

#define rrot(x, n)	ROTR32(x, n)

bool fulltest_pascal(const uint32_t *vhash64, const uint32_t *ptarget)
{
	for(int i = 0; i < 8; i++)
	{
		if(vhash64[i] > ptarget[7 - i])
			return false;
		else
			if(vhash64[i] < ptarget[7 - i])
				return true;
	}
	return true;
}

void pascal_hash(uint32_t *output, const uint32_t *data, uint32_t datasize, uint32_t nonce, const uint32_t *midstate)
{
	int i;
	uint32_t s0, s1, t1, t2, maj, ch, a, b, c, d, e, f, g, h;
	uint32_t w[64];

	const uint32_t k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};
	const uint32_t hc[8] = {
		0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
		0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
	};
	int fullblocks = datasize / 64;
	for(i = 0; i <= 15; i++)
	{
		w[i] = data[i + fullblocks * 16];
	}

	w[(datasize%64-4)/4] = nonce;
	for(i = 16; i <= 63; i++)
	{
		s0 = rrot(w[i - 15], 7) ^ rrot(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = rrot(w[i - 2], 17) ^ rrot(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = midstate[0];
	b = midstate[1];
	c = midstate[2];
	d = midstate[3];
	e = midstate[4];
	f = midstate[5];
	g = midstate[6];
	h = midstate[7];
	for(i = 0; i <= 63; i++)
	{
		s0 = rrot(a, 2) ^ rrot(a, 13) ^ rrot(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		s1 = rrot(e, 6) ^ rrot(e, 11) ^ rrot(e, 25);
		ch = (e & f) ^ ((~e) & g);
		t1 = h + s1 + ch + k[i] + w[i];
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	w[0] = a + midstate[0];
	w[1] = b + midstate[1];
	w[2] = c + midstate[2];
	w[3] = d + midstate[3];
	w[4] = e + midstate[4];
	w[5] = f + midstate[5];
	w[6] = g + midstate[6];
	w[7] = h + midstate[7];
	w[8] = 0x80000000U;
	for(i = 9; i <= 14; i++)
		w[i] = 0U;
	w[15] = 0x0100U;
	for(i = 16; i <= 63; i++)
	{
		s0 = rrot(w[i - 15], 7) ^ rrot(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = rrot(w[i - 2], 17) ^ rrot(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = hc[0];
	b = hc[1];
	c = hc[2];
	d = hc[3];
	e = hc[4];
	f = hc[5];
	g = hc[6];
	h = hc[7];
	for(i = 0; i <= 63; i++)
	{
		s0 = rrot(a, 2) ^ rrot(a, 13) ^ rrot(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		s1 = rrot(e, 6) ^ rrot(e, 11) ^ rrot(e, 25);
		ch = (e & f) ^ ((~e) & g);
		t1 = h + s1 + ch + k[i] + w[i];
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	output[0] = a + hc[0];
	output[1] = b + hc[1];
	output[2] = c + hc[2];
	output[3] = d + hc[3];
	output[4] = e + hc[4];
	output[5] = f + hc[5];
	output[6] = g + hc[6];
	output[7] = h + hc[7];
}

void pascal_midstate(const uint32_t *data, uint32_t *hc)
{
	int i;
	uint32_t s0, s1, t1, t2, maj, ch, a, b, c, d, e, f, g, h;
	uint32_t w[64];

	const uint32_t k[64] = {
		0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
		0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
		0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
		0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
		0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
		0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
		0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
		0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
	};

	for(i = 0; i <= 15; i++)
	{
		w[i] = data[i];
	}
	for(i = 16; i <= 63; i++)
	{
		s0 = ROTR32(w[i - 15], 7) ^ ROTR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = ROTR32(w[i - 2], 17) ^ ROTR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = hc[0];
	b = hc[1];
	c = hc[2];
	d = hc[3];
	e = hc[4];
	f = hc[5];
	g = hc[6];
	h = hc[7];
	for(i = 0; i <= 63; i++)
	{
		s0 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		s1 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25);
		ch = (e & f) ^ ((~e) & g);
		t1 = h + s1 + ch + k[i] + w[i];
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	hc[0] += a;
	hc[1] += b;
	hc[2] += c;
	hc[3] += d;
	hc[4] += e;
	hc[5] += f;
	hc[6] += g;
	hc[7] += h;
}

int scanhash_pascal(int thr_id, uint32_t *pdata, uint32_t datasize,
					uint32_t *ptarget, uint32_t max_nonce,
					uint32_t *hashes_done)
{
	static THREAD uint32_t *result = nullptr;
	static THREAD volatile bool init = false;
	static THREAD uint32_t *pdata2 = nullptr;

	const uint32_t first_nonce = pdata[datasize / 4 - 1];
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, 1U << 28);
	throughput = min(throughput, (max_nonce - first_nonce + 1));

	if(opt_benchmark)
		ptarget[7] = 0x0005;

	if(!init)
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		pascal_cpu_init(thr_id);
		CUDA_SAFE_CALL(cudaMallocHost(&result, 2 * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMallocHost(&pdata2, datasize+64));
		result[0] = 0; result[1] = 0;
		init = true;
	}
	const int chunks = datasize / 64;
	const int datasize32 = datasize / 4;

	// change endianness
	for(size_t i = 0; i < datasize32; i ++)
		pdata2[i] = swab32(pdata[i]);

	// padding overflow
	if(datasize % 64 > 53)
	{
		applog(LOG_ERR, "Error: data size %d is not being supported yet", datasize);
		sleep(1);
		return 0;
	}

	// add padding
	if(datasize % 64 > 0)
	{
		pdata2[datasize32] = 0x80000000;
		for(int i = 1; i <= 16 - datasize32 % 16; i++)
			pdata2[datasize32 + i] = 0;
		pdata2[chunks * 16 + 15] = datasize * 8;
	}

	uint32_t ms[8] =
	{
		0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
		0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
	};

	if(chunks > 0)
		pascal_midstate(pdata2, ms);
	if(chunks > 1)
		pascal_midstate(pdata2 + 16, ms);
	if(chunks > 2)
		pascal_midstate(pdata2 + 32, ms);

	copydata(pdata2 + chunks * 16);
	do
	{
		pascal_cpu_hash(thr_id, throughput, pdata2[datasize32 - 1], (datasize % 64) - 4, ms, result);

		if(stop_mining)
		{
			mining_has_stopped[thr_id] = true;
			pthread_exit(nullptr);
		}
		if(result[0] != 0)
		{
			uint32_t vhash64[8] = {0};
			pascal_hash(vhash64, pdata2, datasize, result[0], ms);

			if(!opt_verify || (vhash64[0] == 0 && fulltest_pascal(vhash64, ptarget)))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata2[datasize32 - 1] - first_nonce + throughput;
				if(result[1] != 0)
				{
					pascal_hash(vhash64, pdata2, datasize, result[1], ms);
					if(!opt_verify || (vhash64[0] == 0 && fulltest_pascal(vhash64, ptarget)))
					{
						pdata[datasize32 + 1] = result[1];
						res++;
						if(opt_benchmark)
							applog(LOG_INFO, "GPU #%d Found second nounce %08x", device_map[thr_id], result[1]);
					}
					else
					{
						if(vhash64[0] > 0)
						{
							applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], result[1]);
						}
					}
				}
				pdata[datasize32 - 1] = result[0];
				if(opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", device_map[thr_id], result[0]);
				return res;
			}
			else
			{
				if(vhash64[0] > 0)
				{
					applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], result[0]);
				}
			}
		}

		pdata2[datasize32 - 1] += throughput;
	} while(!work_restart[thr_id].restart && max_nonce - throughput > pdata2[datasize32 - 1]);

	*hashes_done = pdata2[datasize32 - 1] - first_nonce;

	return 0;
}
