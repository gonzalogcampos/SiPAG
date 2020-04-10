// cudaNoise
// Library of common 3D noise functions for CUDA kernels

#pragma once

#include <cuda_runtime.h>
#include <corecrt_math.h>


// Utility functions

// Hashing function (used for fast on-device pseudorandom numbers for randomness in noise)
__device__ unsigned int hash(unsigned int seed)
{
	seed = (seed + 0x7ed55d16) + (seed << 12);
	seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
	seed = (seed + 0x165667b1) + (seed << 5);
	seed = (seed + 0xd3a2646c) ^ (seed << 9);
	seed = (seed + 0xfd7046c5) + (seed << 3);
	seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

	return seed;
}

// Maps from the signed range [0, 1] to unsigned [-1, 1]
// NOTE: no clamping
__device__ float mapToSigned(float input)
{
	return input * 2.0f - 1.0f;
}


// Random unsigned int for a grid coordinate [0, MAXUINT]
__device__ unsigned int randomIntGrid(float x, float y, float z, float t, float seed = 0.0f)
{
	return hash((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + t * 892.f + 3824 + seed));
}


// Helper functions for noise

// Linearly interpolate between two float values
__device__  float lerp(float a, float b, float ratio)
{
	return a * (1.0f - ratio) + b * ratio;
}


// Fast gradient function for gradient noise
__device__ float grad(int hash, float x, float y, float z, float t)
{
	switch (hash & 0x1f)
	{
	case 0x00: return x + y;
	case 0x01: return -x + y;
	case 0x02: return x - y;
	case 0x03: return -x - y;
	case 0x04: return x + z;
	case 0x05: return -x + z;
	case 0x06: return x - z;
	case 0x07: return -x - z;
	case 0x08: return y + z;
	case 0x09: return -y + z;
	case 0x0A: return y - z;
	case 0x0B: return -y - z;
	case 0x0C: return -y + z;
	case 0x0D: return -y - z;
	case 0x0E: return x + t;
	case 0x0F: return -x + t;
	case 0x10: return x - t;
	case 0x11: return -x - t;
	case 0x12: return y + t;
	case 0x13: return -y + t;
	case 0x14: return y - t;
	case 0x15: return -y - t;
	case 0x16: return z + t;
	case 0x17: return -z + t;
	case 0x18: return z - t;
	case 0x19: return -z - t;
	case 0x1A: return x + y;
	case 0x1B: return -x + y;
	case 0x1C: return x - y;
	case 0x1D: return -x - y;
	case 0x1E: return x + z;
	case 0x1F: return -x + z;
	default: return 0; // never happens
	}
}

// Ken Perlin's fade function for Perlin noise
__device__ float fade(float t)
{
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);     // 6t^5 - 15t^4 + 10t^3
}


// Perlin gradient noise
__device__ float perlinNoise(float3 pos, float t, float scale, int seed)
{
	float fseed = (float)seed;

	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

	// zero corner integer position
	float ix = floorf(pos.x);
	float iy = floorf(pos.y);
	float iz = floorf(pos.z);
	float it = floorf(t);

	// current position within unit cube
	pos.x -= ix;
	pos.y -= iy;
	pos.z -= iz;
	t -= it;

	// adjust for fade
	float u = fade(pos.x);
	float v = fade(pos.y);
	float w = fade(pos.z);
	float k = fade(t);

	// influence values
	float i0000 = grad(randomIntGrid(ix, iy, iz, t, fseed), pos.x, pos.y, pos.z, t);
	float i1000 = grad(randomIntGrid(ix + 1.0f, iy, iz, t, fseed), pos.x - 1.0f, pos.y, pos.z, t);
	float i0100 = grad(randomIntGrid(ix, iy + 1.0f, iz, t, fseed), pos.x, pos.y - 1.0f, pos.z, t);
	float i1100 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz, t, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z, t);
	float i0010 = grad(randomIntGrid(ix, iy, iz + 1.0f, t, fseed), pos.x, pos.y, pos.z - 1.0f, t);
	float i1010 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f, t, fseed), pos.x - 1.0f, pos.y, pos.z - 1.0f, t);
	float i0110 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f, t, fseed), pos.x, pos.y - 1.0f, pos.z - 1.0f, t);
	float i1110 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz + 1.0f, t, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f, t);
	float i0001 = grad(randomIntGrid(ix, iy, iz, t + 1.0f, fseed), pos.x, pos.y, pos.z, t - 1.0f);
	float i1001 = grad(randomIntGrid(ix + 1.0f, iy, iz, t + 1.0f, fseed), pos.x - 1.0f, pos.y, pos.z, t - 1.0f);
	float i0101 = grad(randomIntGrid(ix, iy + 1.0f, iz, t + 1.0f, fseed), pos.x, pos.y - 1.0f, pos.z, t - 1.0f);
	float i1101 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz, t + 1.0f, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z, t - 1.0f);
	float i0011 = grad(randomIntGrid(ix, iy, iz + 1.0f, t + 1.0f, fseed), pos.x, pos.y, pos.z - 1.0f, t - 1.0f);
	float i1011 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f, t + 1.0f, fseed), pos.x - 1.0f, pos.y, pos.z - 1.0f, t - 1.0f);
	float i0111 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f, t + 1.0f, fseed), pos.x, pos.y - 1.0f, pos.z - 1.0f, t - 1.0f);
	float i1111 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz + 1.0f, t + 1.0f, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f, t - 1.0f);

	// interpolation
	float x000 = lerp(i0000, i1000, u);
	float x100 = lerp(i0100, i1100, u);
	float x010 = lerp(i0010, i1010, u);
	float x110 = lerp(i0110, i1110, u);
	float x001 = lerp(i0001, i1001, u);
	float x101 = lerp(i0101, i1101, u);
	float x011 = lerp(i0011, i1011, u);
	float x111 = lerp(i0111, i1111, u);

	float y00 = lerp(x000, x100, v);
	float y10 = lerp(x010, x110, v);
	float y01 = lerp(x001, x101, v);
	float y11 = lerp(x011, x111, v);

	float z0 = lerp(y00, y10, w);
	float z1 = lerp(y01, y11, w);

	float avg = lerp(z0, z1, k);

	return avg;
}

// Derived noise functions

// Fast function for fBm using perlin noise
__device__ float repeaterPerlin(float3 pos, float time, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), time, 1.0f, seed * (i + 3)) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

// Fast function for fBm using perlin absolute noise
// Originally called "turbulence", this method takes the absolute value of each octave before adding
__device__ float repeaterPerlinAbs(float3 pos, float time, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += fabsf(perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), time, 1.0f, seed)) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	// Map the noise back to the standard expected range [-1, 1]
	return mapToSigned(acc);
}

// Generic turbulence function
// Uses a first pass of noise to offset the input vectors for the second pass
__device__ float turbulence(float3 pos, float time, float scaleIn, float scaleOut, int seed, float strength)
{

	pos.x += perlinNoise(pos, time, scaleIn, seed ^ 0x74827384) * strength;
	pos.y += perlinNoise(pos, time, scaleIn, seed ^ 0x10938478) * strength;
	pos.z += perlinNoise(pos, time, scaleIn, seed ^ 0x62723883) * strength;

	return perlinNoise(pos, time, scaleOut, seed);

	return 0.0f;
}

// Turbulence using repeaters for the first and second pass
__device__ float repeaterTurbulence(float3 pos, float time, float scaleIn, float scaleOut, int seed, float strength, int n)
{
	pos.x += (repeaterPerlin(make_float3(pos.x, pos.y, pos.z), time, scaleIn, seed ^ 0x41728394, n, 2.0f, 0.5f)) * strength;
	pos.y += (repeaterPerlin(make_float3(pos.x, pos.y, pos.z), time, scaleIn, seed ^ 0x72837263, n, 2.0f, 0.5f)) * strength;
	pos.z += (repeaterPerlin(make_float3(pos.x, pos.y, pos.z), time, scaleIn, seed ^ 0x26837363, n, 2.0f, 0.5f)) * strength;

	return repeaterPerlin(pos, time, scaleOut, seed ^ 0x3f821dab, n, 2.0f, 0.5f);
}