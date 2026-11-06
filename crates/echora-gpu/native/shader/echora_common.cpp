// SPDX-License-Identifier: GPL-3.0-or-later
// The few Eden common-library services the shader recompiler and its settings
// reference, without Eden's logging backend, filesystem layout or time zones.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/assert.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/time_zone.h"

namespace Common::Log {
void FmtLogMessageImpl(Class, Level level, const char* filename, unsigned int line,
                       const char*, fmt::string_view format, const fmt::format_args& args) {
    // Warnings and worse reach stderr; translation detail stays quiet.
    if (level < Level::Warning || std::getenv("ECHORA_SHADER_QUIET")) return;
    std::fprintf(stderr, "shader: %s:%u: %s\n", filename, line, fmt::vformat(format, args).c_str());
}
} // namespace Common::Log

void AssertFailSoftImpl() {}

[[noreturn]] void AssertFatalImpl() {
    // Recompiler asserts become a failed translation instead of an abort.
    throw std::logic_error("shader recompiler assertion");
}

namespace Common::FS {
std::string PathToUTF8String(const std::filesystem::path& path) {
    return path.string();
}
const std::filesystem::path& GetEdenPath(EdenPath) {
    static const std::filesystem::path empty;
    return empty;
}
std::string GetEdenPathString(EdenPath) {
    return {};
}
} // namespace Common::FS

namespace Common::TimeZone {
const std::array<const char*, 46>& GetTimeZoneStrings() {
    static const std::array<const char*, 46> zones{};
    return zones;
}
std::string FindSystemTimeZone() {
    return "UTC";
}
} // namespace Common::TimeZone
