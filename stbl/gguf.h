// Minimal GGUF reader for STBLarge's converted DeepSeek-V4.1 file: mmap, metadata, tensor table.
#pragma once
#include <stdint.h>

#include <map>
#include <string>
#include <vector>

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> dims;  // GGUF order: dims[0] is the contiguous (innermost) dimension
    uint64_t offset = 0;         // absolute file offset of the data
    uint64_t nbytes = 0;
};

struct GgufFile {
    int fd = -1;
    const uint8_t *map = nullptr;
    uint64_t size = 0;
    std::map<std::string, std::pair<uint32_t, uint64_t>> kv;  // key -> (type, file offset of value)
    std::map<std::string, GgufTensor> tensors;

    bool open(const char *path, std::string &err);
    ~GgufFile();

    const GgufTensor *tensor(const std::string &name) const;
    uint32_t u32(const std::string &key) const;
    float f32(const std::string &key) const;
    std::vector<uint64_t> uints(const std::string &key) const;  // u32 or u64 array
    const uint8_t *data(const GgufTensor &t) const { return map + t.offset; }
};

enum : uint32_t {
    QT_F32 = 0, QT_I8 = 24, QT_BF16 = 30, QT_MXFP4 = 39, QT_STBL_E4M3 = 200, QT_STBL_E8M0 = 201,
};
