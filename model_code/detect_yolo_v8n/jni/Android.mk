########### build libnn_yolo_v8n
LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)

common_src_files := vnn_yolov8n.c yolov8n_process.c yolo_v8n.c 

common_c_includes := \
    $(LOCAL_PATH)/applib-ovxinc-include \
	$(LOCAL_PATH)/include \
	$(LOCAL_PATH)/service-ovx_inc \
	system/core/libutils/include 
	



LOCAL_SRC_FILES := $(common_src_files)

LOCAL_C_INCLUDES := \
  $(common_c_includes) 

  
CFLAG+=-Wno-error=unused-variable
CFLAG+=-Wno-error=format
CFLAG+=-Wno-error=unused-function
CFLAG+=-Wno-error=unused-parameter
CFLAG+=-Wno-error=macro-redefined  
CFLAG+=-Wno-error=sign-compare
CFLAG+=-Wno-error=typedef-redefinition
LOCAL_CFLAGS += $(CFLAG)


LOCAL_LDLIBS +=  -L$(LOCAL_PATH)/libs -lovxlib 

LOCAL_MODULE := libnn_yolo_v8n


ifeq ($(shell test $(PLATFORM_SDK_VERSION) -ge 26 && echo OK),OK)
LOCAL_PROPRIETARY_MODULE := true
endif

LOCAL_PRELINK_MODULE := false
LOCAL_MODULE_TAGS := optional
LOCAL_VENDOR_MODULE := true


include $(BUILD_SHARED_LIBRARY)
