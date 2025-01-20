#ifndef __YOLO_V3_TINY_PROCESS__
#define __YOLO_V3_TINY_PROCESS__

#include "vnn_global.h"
#include "nn_detect_common.h"

typedef unsigned char   uint8_t;
typedef unsigned int   uint32_t;

void yolov3_tiny_preprocess(input_image_t imageData, uint8_t *ptr);
void yolov3_tiny_postprocess(vsi_nn_graph_t *graph, pDetResult resultData);

#endif
