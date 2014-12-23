
#ifndef GPUTUPLE_H
#define GPUTUPLE_H

#include <GPUetc/common/GNValue.h>


namespace voltdb{

//configuration for non-index join
#define BLOCK_SIZE_X 320  //outer ,left
//#define BLOCK_SIZE_Y 256  //inner ,right

//configuration for single hash join
#define PARTITION 64
#define RADIX 6
#define PART_C_NUM 16
#define SHARED_MAX PARTITION * PART_C_NUM

#define RIGHT_PER_TH 256

#define PART_STANDARD 1
#define JOIN_SHARED 256


//structure when transporting data to GPU.
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
