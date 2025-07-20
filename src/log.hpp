#pragma once

#include <cstdio>

#define LOG(_Fmt, ...)  (std::fprintf(stdout, __FUNCTION__ " " _Fmt "\n", ##__VA_ARGS__))
#define ELOG(_Fmt, ...) (std::fprintf(stderr, "%s " __FUNCTION__ " " _Fmt "\n", "[ERROR] ", ##__VA_ARGS__))
