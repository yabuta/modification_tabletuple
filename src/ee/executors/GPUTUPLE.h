
#ifndef GPUTUPLE_H
#define GPUTUPLE_H

#include <GPUetc/common/GNValue.h>


namespace voltdb{

//1blockでのスレッド数の定義。
#define BLOCK_SIZE_X 256  //outer ,left
#define BLOCK_SIZE_Y 128  //inner ,right

#define PARTITION 64
#define RADIX 6
#define PART_C_NUM 16
#define SHARED_MAX PARTITION * PART_C_NUM

#define RIGHT_PER_TH 256

#define PART_STANDARD 1
#define JOIN_SHARED 256


typedef struct _TUPLE {
    int key;
    int val;
} TUPLE;

typedef struct _RESULT {
    int lkey;
    int rkey;
} RESULT;

typedef struct _COLUMNDATA{
    GNValue gn;
    int num;
} COLUMNDATA;


}

#endif
