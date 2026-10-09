#include "graphics/host_gpu/renderer/image/clearRegister.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

using namespace Libs::Graphics;

void Check(bool condition, const char* description) {
	if (!condition) {
		std::fprintf(stderr, "ClearRegisterDecodeTests: failed: %s\n", description);
		std::abort();
	}
}

bool Near(float left, float right) {
	return std::fabs(left - right) <= 1e-6f * std::max(1.0f, std::fabs(right));
}

void CheckFloat4(const vk::ClearColorValue& clear, float r, float g, float b, float a,
                 const char* description) {
	Check(Near(clear.float32[0], r) && Near(clear.float32[1], g) && Near(clear.float32[2], b) &&
	          Near(clear.float32[3], a),
	      description);
}

void TestMetadataEqualityIncludesWord1() {
	ImageMetadataInfo low {};
	ImageMetadataInfo high {};
	low.dcc_clear_register_valid  = true;
	high.dcc_clear_register_valid = true;
	high.dcc_clear_word1          = 1;
	Check(!(low == high), "metadata equality includes CLEAR_WORD1");
}

void TestRgba16FloatWordsAndOptIn() {
	vk::ClearColorValue clear {};
	// R=1, G=-2, B=0.5, A=0, with the final two channels in CLEAR_WORD1.
	Check(!DecodeColorRegisterClear(vk::Format::eR16G16B16A16Sfloat, 0xc0003c00u, 0x00003800u,
	                                false, clear),
	      "legacy path rejects RGBA16F register clear");
	Check(DecodeColorRegisterClear(vk::Format::eR16G16B16A16Sfloat, 0xc0003c00u, 0x00003800u,
	                               true, clear),
	      "opt-in RGBA16F register clear decodes");
	CheckFloat4(clear, 1.0f, -2.0f, 0.5f, 0.0f, "RGBA16F channels use both words");
	Check(DecodeColorRegisterClear(vk::Format::eR16G16B16A16Sfloat, 0, 0, true, clear),
	      "RGBA16F zero clear decodes");
	CheckFloat4(clear, 0.0f, 0.0f, 0.0f, 0.0f, "RGBA16F zero clear is all zero");
}

void TestWideScalarAndPackedFormats() {
	vk::ClearColorValue clear {};
	Check(DecodeColorRegisterClear(vk::Format::eR16G16Sfloat, 0x3c00bc00u, 0, true, clear),
	      "RG16F decodes");
	CheckFloat4(clear, -1.0f, 1.0f, 0.0f, 0.0f, "RG16F channels");
	Check(DecodeColorRegisterClear(vk::Format::eR16Unorm, 0x0000ffffu, 0, true, clear),
	      "R16 UNORM decodes");
	CheckFloat4(clear, 1.0f, 0.0f, 0.0f, 0.0f, "R16 UNORM value");
	Check(DecodeColorRegisterClear(vk::Format::eR8G8Unorm, 0x0000ff40u, 0, true, clear),
	      "RG8 UNORM decodes");
	Check(Near(clear.float32[0], 64.0f / 255.0f) && Near(clear.float32[1], 1.0f),
	      "RG8 UNORM channel order");
	const uint32_t packed_ones = 0x3c0u | (0x3c0u << 11u) | (0x1e0u << 22u);
	Check(DecodeColorRegisterClear(vk::Format::eB10G11R11UfloatPack32, packed_ones, 0, true,
	                               clear),
	      "B10G11R11 decodes");
	CheckFloat4(clear, 1.0f, 1.0f, 1.0f, 1.0f, "B10G11R11 packed channel order");
}

void TestAllHalfFloatPatterns() {
	for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
		vk::ClearColorValue clear {};
		Check(DecodePackedColorClear64(vk::Format::eR16Sfloat, bits, 0, clear),
		      "all half-float encodings decode");
		const auto exponent = (bits >> 10u) & 31u;
		const auto mantissa = bits & 1023u;
		const auto negative = (bits & 0x8000u) != 0;
		if (exponent == 31u) {
			Check(mantissa != 0 ? std::isnan(clear.float32[0]) : std::isinf(clear.float32[0]),
			      "half infinity and NaN classes preserved");
		} else {
			const auto magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -24)
			    : std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
			                 static_cast<int>(exponent) - 15);
			Check(clear.float32[0] == (negative ? -magnitude : magnitude),
			      "normal and subnormal half values preserved");
		}
		Check(std::signbit(clear.float32[0]) == negative, "half sign preserved including zero");
	}
}

void TestWideIntegerAndFloatWords() {
	vk::ClearColorValue clear {};
	Check(DecodeColorRegisterClear(vk::Format::eR32G32Uint, 0x12345678u, 0x9abcdef0u, true,
	                               clear),
	      "RG32 UINT decodes");
	Check(clear.uint32[0] == 0x12345678u && clear.uint32[1] == 0x9abcdef0u,
	      "RG32 UINT preserves both words");
	Check(DecodeColorRegisterClear(vk::Format::eR32G32Sfloat, 0x3f800000u, 0xc0000000u, true,
	                               clear),
	      "RG32 SFLOAT decodes");
	Check(Near(clear.float32[0], 1.0f) && Near(clear.float32[1], -2.0f),
	      "RG32 SFLOAT preserves both words");
}

void TestLegacyFormatsAndSwizzles() {
	for (const bool wide_enabled: {false, true}) {
		vk::ClearColorValue clear {};
		Check(DecodeColorRegisterClear(vk::Format::eR8G8B8A8Unorm, 0xff0000ffu, 0x12345678u,
		                               wide_enabled, clear),
		      "RGBA8 remains decodable");
		CheckFloat4(clear, 1.0f, 0.0f, 0.0f, 1.0f, "RGBA8 keeps legacy lane order");
		Check(DecodeColorRegisterClear(vk::Format::eB8G8R8A8Unorm, 0xff0000ffu, 0xffffffffu,
		                               wide_enabled, clear),
		      "BGRA8 remains decodable");
		CheckFloat4(clear, 0.0f, 0.0f, 1.0f, 1.0f, "BGRA8 keeps legacy swizzle");
		Check(DecodeColorRegisterClear(vk::Format::eR8G8B8A8Srgb, 0xff0000ffu, 0xffffffffu,
		                               wide_enabled, clear),
		      "RGBA8 sRGB remains decodable");
		CheckFloat4(clear, 1.0f, 0.0f, 0.0f, 1.0f, "RGBA8 sRGB keeps legacy lane order");
		Check(DecodeColorRegisterClear(vk::Format::eR32Uint, 0x89abcdefu, 0xffffffffu,
		                               wide_enabled, clear) && clear.uint32[0] == 0x89abcdefu,
		      "R32 UINT stays word0-only");
	}
	vk::ClearColorValue clear {};
	Check(!DecodeColorRegisterClear(vk::Format::eBc1RgbaUnormBlock, 0, 0, true, clear),
	      "block-compressed formats remain unsupported");
}

} // namespace

int main() {
	TestMetadataEqualityIncludesWord1();
	TestRgba16FloatWordsAndOptIn();
	TestWideScalarAndPackedFormats();
	TestAllHalfFloatPatterns();
	TestWideIntegerAndFloatWords();
	TestLegacyFormatsAndSwizzles();
	std::puts("ClearRegisterDecodeTests: all passed");
	return 0;
}
