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
__device__ unsigned int randomIntGrid(float x, float y, float z, float seed = 0.0f)
{
	return hash((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + 3824 + seed));
}


// Helper functions for noise

// Linearly interpolate between two float values
__device__  float lerp(float a, float b, float ratio)
{
	return a * (1.0f - ratio) + b * ratio;
}


// Fast gradient function for gradient noise
__device__ float grad(int hash, float x, float y, float z)
{
	switch (hash & 0xF)
	{
	case 0x0: return x + y;
	case 0x1: return -x + y;
	case 0x2: return x - y;
	case 0x3: return -x - y;
	case 0x4: return x + z;
	case 0x5: return -x + z;
	case 0x6: return x - z;
	case 0x7: return -x - z;
	case 0x8: return y + z;
	case 0x9: return -y + z;
	case 0xA: return y - z;
	case 0xB: return -y - z;
	case 0xC: return y + x;
	case 0xD: return -y + z;
	case 0xE: return y - x;
	case 0xF: return -y - z;
	default: return 0; // never happens
	}
}

// Ken Perlin's fade function for Perlin noise
__device__ float fade(float t)
{
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);     // 6t^5 - 15t^4 + 10t^3
}


// Perlin gradient noise
__device__ float perlinNoise(float3 pos, float scale, int seed)
{
	float fseed = (float)seed;

	pos.x = pos.x * scale;
	pos.y = pos.y * scale;
	pos.z = pos.z * scale;

	// zero corner integer position
	float ix = floorf(pos.x);
	float iy = floorf(pos.y);
	float iz = floorf(pos.z);

	// current position within unit cube
	pos.x -= ix;
	pos.y -= iy;
	pos.z -= iz;

	// adjust for fade
	float u = fade(pos.x);
	float v = fade(pos.y);
	float w = fade(pos.z);

	// influence values
	float i000 = grad(randomIntGrid(ix, iy, iz, fseed), pos.x, pos.y, pos.z);
	float i100 = grad(randomIntGrid(ix + 1.0f, iy, iz, fseed), pos.x - 1.0f, pos.y, pos.z);
	float i010 = grad(randomIntGrid(ix, iy + 1.0f, iz, fseed), pos.x, pos.y - 1.0f, pos.z);
	float i110 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z);
	float i001 = grad(randomIntGrid(ix, iy, iz + 1.0f, fseed), pos.x, pos.y, pos.z - 1.0f);
	float i101 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f, fseed), pos.x - 1.0f, pos.y, pos.z - 1.0f);
	float i011 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f, fseed), pos.x, pos.y - 1.0f, pos.z - 1.0f);
	float i111 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz + 1.0f, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);

	// interpolation
	float x00 = lerp(i000, i100, u);
	float x10 = lerp(i010, i110, u);
	float x01 = lerp(i001, i101, u);
	float x11 = lerp(i011, i111, u);

	float y0 = lerp(x00, x10, v);
	float y1 = lerp(x01, x11, v);

	float avg = lerp(y0, y1, w);

	return avg;
}

// Derived noise functions

// Fast function for fBm using perlin noise
__device__ float repeaterPerlin(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed * (i + 3)) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	return acc;
}

// Fast function for fBm using perlin absolute noise
// Originally called "turbulence", this method takes the absolute value of each octave before adding
__device__ float repeaterPerlinAbs(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += fabsf(perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f, seed)) * amp;
		scale *= lacunarity;
		amp *= decay;
	}

	// Map the noise back to the standard expected range [-1, 1]
	return mapToSigned(acc);
}

// Generic turbulence function
// Uses a first pass of noise to offset the input vectors for the second pass
__device__ float turbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength)
{

	pos.x += perlinNoise(pos, scaleIn, seed ^ 0x74827384) * strength;
	pos.y += perlinNoise(pos, scaleIn, seed ^ 0x10938478) * strength;
	pos.z += perlinNoise(pos, scaleIn, seed ^ 0x62723883) * strength;

	return perlinNoise(pos, scaleOut, seed);

	return 0.0f;
}

// Turbulence using repeaters for the first and second pass
__device__ float repeaterTurbulence(float3 pos, float scaleIn, float scaleOut, int seed, float strength, int n)
{
	pos.x += (repeaterPerlin(make_float3(pos.x, pos.y, pos.z), scaleIn, seed ^ 0x41728394, n, 2.0f, 0.5f)) * strength;
	pos.y += (repeaterPerlin(make_float3(pos.x, pos.y, pos.z), scaleIn, seed ^ 0x72837263, n, 2.0f, 0.5f)) * strength;
	pos.z += (repeaterPerlin(make_float3(pos.x, pos.y, pos.z), scaleIn, seed ^ 0x26837363, n, 2.0f, 0.5f)) * strength;

	return repeaterPerlin(pos, scaleOut, seed ^ 0x3f821dab, n, 2.0f, 0.5f);
}