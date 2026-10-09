#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_CLEARREGISTER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_CLEARREGISTER_H_

#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <cstdlib>
#include <limits>

namespace Libs::Graphics {

// CLEAR_WORD1 decoding stays opt-in until the affected games are validated.
[[nodiscard]] inline bool ClearRegisterWideEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CLEAR_REGISTER_WIDE");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

// Decode raw CB_COLORn_CLEAR_WORD0/1 texel bits. Existing 32-bit formats deliberately fall through
// to DecodePackedColorClear so their channel order, sRGB conversion, and packed swizzles stay exact.
[[nodiscard]] inline bool DecodePackedColorClear64(vk::Format format, uint32_t word0,
                                                   uint32_t word1, vk::ClearColorValue& clear) {
	const auto half_to_float = [](uint32_t bits) {
		const uint32_t sign     = (bits & 0x8000u) << 16u;
		const uint32_t exponent = (bits >> 10u) & 0x1fu;
		uint32_t       mantissa = bits & 0x3ffu;
		if (exponent == 0x1fu) {
			return std::bit_cast<float>(sign | 0x7f800000u | (mantissa << 13u));
		}
		if (exponent != 0) {
			return std::bit_cast<float>(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
		}
		if (mantissa == 0) {
			return std::bit_cast<float>(sign);
		}
		uint32_t shift = 0;
		while ((mantissa & 0x400u) == 0) {
			mantissa <<= 1u;
			++shift;
		}
		mantissa &= 0x3ffu;
		return std::bit_cast<float>(sign | ((113u - shift) << 23u) | (mantissa << 13u));
	};
	const std::array<uint16_t, 4> halves {static_cast<uint16_t>(word0),
	                                      static_cast<uint16_t>(word0 >> 16u),
	                                      static_cast<uint16_t>(word1),
	                                      static_cast<uint16_t>(word1 >> 16u)};
	vk::ClearColorValue next {};
	switch (format) {
		case vk::Format::eR16G16B16A16Sfloat:
			for (uint32_t i = 0; i < 4; ++i) {
				next.float32[i] = half_to_float(halves[i]);
			}
			break;
		case vk::Format::eR16G16B16A16Unorm:
			for (uint32_t i = 0; i < 4; ++i) {
				next.float32[i] = static_cast<float>(halves[i]) / 65535.0f;
			}
			break;
		case vk::Format::eR16G16B16A16Uint:
			for (uint32_t i = 0; i < 4; ++i) {
				next.uint32[i] = halves[i];
			}
			break;
		case vk::Format::eR16G16B16A16Sint:
			for (uint32_t i = 0; i < 4; ++i) {
				next.int32[i] = std::bit_cast<int16_t>(halves[i]);
			}
			break;
		case vk::Format::eR32G32Sfloat:
			next.float32[0] = std::bit_cast<float>(word0);
			next.float32[1] = std::bit_cast<float>(word1);
			break;
		case vk::Format::eR32G32Uint:
			next.uint32[0] = word0;
			next.uint32[1] = word1;
			break;
		case vk::Format::eR32G32Sint:
			next.int32[0] = std::bit_cast<int32_t>(word0);
			next.int32[1] = std::bit_cast<int32_t>(word1);
			break;
		case vk::Format::eR16Sfloat: next.float32[0] = half_to_float(halves[0]); break;
		case vk::Format::eR16G16Sfloat:
			next.float32[0] = half_to_float(halves[0]);
			next.float32[1] = half_to_float(halves[1]);
			break;
		case vk::Format::eR16Unorm: next.float32[0] = static_cast<float>(halves[0]) / 65535.0f; break;
		case vk::Format::eR16G16Unorm:
			next.float32[0] = static_cast<float>(halves[0]) / 65535.0f;
			next.float32[1] = static_cast<float>(halves[1]) / 65535.0f;
			break;
		case vk::Format::eR8Unorm:
			next.float32[0] = static_cast<float>(word0 & 0xffu) / 255.0f;
			break;
		case vk::Format::eR8G8Unorm:
			next.float32[0] = static_cast<float>(word0 & 0xffu) / 255.0f;
			next.float32[1] = static_cast<float>((word0 >> 8u) & 0xffu) / 255.0f;
			break;
		case vk::Format::eB10G11R11UfloatPack32: {
			const auto decode_component = [](uint32_t bits, uint32_t mantissa_bits) {
				const uint32_t mantissa = bits & ((1u << mantissa_bits) - 1u);
				const uint32_t exponent = bits >> mantissa_bits;
				if (exponent == 0x1fu) {
					return mantissa != 0 ? std::numeric_limits<float>::quiet_NaN()
					                    : std::numeric_limits<float>::infinity();
				}
				const auto fraction = static_cast<float>(mantissa) /
				                      static_cast<float>(1u << mantissa_bits);
				return exponent == 0 ? std::ldexp(fraction, -14)
				                     : std::ldexp(1.0f + fraction, static_cast<int>(exponent) - 15);
			};
			next.float32[0] = decode_component(word0 & 0x7ffu, 6);
			next.float32[1] = decode_component((word0 >> 11u) & 0x7ffu, 6);
			next.float32[2] = decode_component(word0 >> 22u, 5);
			next.float32[3] = 1.0f;
			break;
		}
		default: return DecodePackedColorClear(format, word0, clear);
	}
	clear = next;
	return true;
}

[[nodiscard]] inline bool DecodeColorRegisterClear(vk::Format format, uint32_t word0,
                                                   uint32_t word1, bool wide_enabled,
                                                   vk::ClearColorValue& clear) {
	return wide_enabled ? DecodePackedColorClear64(format, word0, word1, clear)
	                    : DecodePackedColorClear(format, word0, clear);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_CLEARREGISTER_H_
