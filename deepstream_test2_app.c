/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "nvds_yml_parser.h"

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "nvds_tracker_meta.h"

#define PGIE_CONFIG_FILE  "dstest2_pgie_config.txt"
#define SGIE1_CONFIG_FILE "dstest2_sgie1_config.txt"
#define SGIE2_CONFIG_FILE "dstest2_sgie2_config.txt"
#define MAX_DISPLAY_LEN 64

#define TRACKER_CONFIG_FILE "dstest2_tracker_config.txt"
#define MAX_TRACKING_ID_LEN 16

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Create an inference element instance of the specified type. */
#define CREATE_GIE_INSTANCE(element, type, name) \
  if ((type) == NVDS_GIE_PLUGIN_INFER_SERVER) { \
    element = gst_element_factory_make ("nvinferserver", name); \
  } else { \
    element = gst_element_factory_make ("nvinfer", name); \
  }

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

gint frame_number = 0;
/* These are the strings of the labels for the respective models */
gchar sgie1_classes_str[20][32] =
    { "Acura", "Audi", "BMW", "Chevrolet", "Chrysler",
  "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia",
      "Lexus", "Mazda", "Mercedes", "Nissan",
  "Subaru", "Toyota", "Volkswagen"
};

gchar sgie2_classes_str[6][32] = { "coupe", "largevehicle", "sedan", "suv",
  "truck", "van"
};

gchar pgie_classes_str[4][32] =
    { "Vehicle", "TwoWheeler", "Person", "RoadSign" };

/* gie_unique_id is one of the properties in the above dstest2_sgiex_config.txt
 * files. These should be unique and known when we want to parse the Metadata
 * respective to the sgie labels. Ideally these should be read from the config
 * files but for brevity we ensure they are same. */

guint sgie1_unique_id = 2;
guint sgie2_unique_id = 3;

/* This is the buffer probe function that we have registered on the sink pad
 * of the OSD element. All the infer elements in the pipeline shall attach
 * their metadata to the GstBuffer, here we will iterate & process the metadata
 * forex: class ids to strings, counting of class_id objects etc. */
static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsUserMetaList * l_obj_user = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            guint64 id = obj_meta->object_id;

            for (l_obj_user = obj_meta->obj_user_meta_list; l_obj_user != NULL; l_obj_user = l_obj_user->next) {
              NvDsUserMeta *user_meta = (NvDsUserMeta *) l_obj_user->data;             

              if (user_meta && user_meta->base_meta.meta_type == NVDS_TRACKER_OBJ_REID_META && user_meta->user_meta_data) {
                NvDsObjReid *pReidObj = (NvDsObjReid *) (user_meta->user_meta_data);

                // if (pReidObj != NULL && pReidObj->ptr_host != NULL && pReidObj->featureSize > 0) {
                //   printf ("REID embedding for Object %lu [", id);
                //   for (guint ele_i = 0; ele_i < pReidObj->featureSize; ele_i++) {
                //     printf ("%f, ", pReidObj->ptr_host[ele_i]);
                //   }
                //   printf ("]\n");
                // }

                // if (pReidObj != NULL && pReidObj->ptr_host != NULL && pReidObj->featureSize > 0) {
                //   // Create the directory path
                //   char dir_path[PATH_MAX];
                //   g_snprintf(dir_path, sizeof(dir_path) - 1, "reid_embeddings/%lu", id);
                  
                //   // Ensure directory exists
                //   if (g_mkdir_with_parents(dir_path, 0755) == -1) {
                //       printf("Error: Failed to create directory %s: %s (errno: %d)\n", dir_path, strerror(errno), errno);
                //   } else {
                //       // Format the full file path
                //       char reid_file[PATH_MAX];
                //       g_snprintf(reid_file, sizeof(reid_file) - 1, "reid_embeddings/%lu/%06lu.txt", 
                //                  id, (gulong) frame_meta->frame_num);
                      
                //       // Open the file for writing
                //       FILE *reid_params_dump_file = fopen(reid_file, "w");
                //       if (!reid_params_dump_file) {
                //           printf("Unable to open file %s: %s (errno: %d)\n", reid_file, strerror(errno), errno);
                //       } else {
                //           // Write the vector data to the file
                //           for (guint ele_i = 0; ele_i < pReidObj->featureSize; ele_i++) {
                //               fprintf(reid_params_dump_file, "%f", pReidObj->ptr_host[ele_i]);
                              
                //               // Add a comma after each element except the last one
                //               if (ele_i < pReidObj->featureSize - 1) {
                //                   fprintf(reid_params_dump_file, ", ");
                //               }
                //           }
                //           fprintf(reid_params_dump_file, "\n");
                          
                //           // Close the file
                //           fclose(reid_params_dump_file);
                //           printf("Successfully saved REID embedding for Object %lu to %s\n", id, reid_file);
                //       }
                //   }
                // }
                
              }
            }
        }
    }
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strrstr (name, "source") == name) {
        g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
  }

}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}


int main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL, *tiler = NULL, *sgie1 = NULL, *sgie2 = NULL, *nvtracker = NULL, *nvdslogger = NULL;
  
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL;
  guint i =0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER, sgie1_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie2_type = NVDS_GIE_PLUGIN_INFER;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    g_printerr ("OR: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
      g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
          "primary-gie"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&sgie1_type, argv[1],
          "secondary-gie1"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&sgie2_type, argv[1],
          "secondary-gie2"));
  }

  /* Create gstreamer elements */

  /* Create Pipeline element that will be a container of other elements */
  pipeline = gst_pipeline_new ("dstest2-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  gst_bin_add (GST_BIN (pipeline), streammux);

  GList *src_list = NULL ;

  if (yaml_config) {

    RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));

    GList * temp = src_list;
    while(temp) {
      num_sources++;
      temp=temp->next;
    }
    g_list_free(temp);
  } else {
      num_sources = argc - 1;
  }

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    GstElement *source_bin= NULL;
    if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {
      g_print("Now playing : %s\n",(char*)(src_list)->data);
      source_bin = create_source_bin (i, (char*)(src_list)->data);
    } else {
      source_bin = create_source_bin (i, argv[i + 1]);
    }
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);

    if (yaml_config) {
      src_list = src_list->next;
    }
  }

  if (yaml_config) {
    g_list_free(src_list);
  }

  /* Use nvinfer or nvinferserver to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  CREATE_GIE_INSTANCE(pgie, pgie_type, "primary-nvinference-engine");

  /* We need to have a tracker to track the identified objects */
  nvtracker = gst_element_factory_make ("nvtracker", "tracker");

  /* We need three secondary gies so lets create 3 more instances of
     nvinfer or nvinferserver */
  CREATE_GIE_INSTANCE(sgie1, sgie1_type, "secondary1-nvinference-engine");
  CREATE_GIE_INSTANCE(sgie2, sgie2_type, "secondary2-nvinference-engine");

  /* Use nvdslogger for perf measurement. */
  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");
  g_object_set (G_OBJECT (nvdslogger), "fps-measurement-interval-sec", 1, NULL);

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if(prop.integrated) {
    sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!pgie || !nvtracker || !sgie1 || !sgie2 || !nvdslogger || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1], "streammux"));

    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(sgie1, argv[1], "secondary-gie1"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(sgie2, argv[1], "secondary-gie2"));
    
    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
      g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    }

    RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1],"osd"));

    RETURN_ON_PARSER_ERROR(nvds_parse_tracker(nvtracker, argv[1], "tracker"));

    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);

    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  gst_bin_add_many (GST_BIN (pipeline), pgie, nvtracker, sgie1, sgie2, nvdslogger,
      nvvidconv, nvosd, tiler, sink, NULL);

  /* Link the elements together */
  if (!gst_element_link_many (streammux, pgie, nvtracker, sgie1,
      sgie2, nvdslogger, tiler, nvvidconv, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Using file: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Iterate */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
