/**
 * @file    filter.cpp
 * @brief   AXI‑Stream HLS kernel implementing three parallel 3×3 convolution filters
 *          on separate color channels (R, G, B).
 *
 * Interfaces:
 *   - AXI‑Lite slave (“control” bundle) for parameters:
 *       · image_width, image_height
 *       · kernel_factor
 *       · kernel[3][3]
 *   - AXI‑Stream inputs/outputs for each channel:
 *       · input_stream  → output_stream
 *       · input_stream2 → output_stream2
 *       · input_stream3 → output_stream3
 *
 * Authors: Zichu Pan, Yifei Li, Weikai Rao
 * Date:   2025-04-20
 */

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"

// maximum width of image (frame)
#define MAX_WIDTH 1920

// 8 bit AXI‑Stream word for a single pixel in single channel
typedef ap_axis<8, 0, 0, 0> AXI_PIXEL;

static void store_channel(
    const AXI_PIXEL &pix,
    int row_idx, int col,
    ap_uint<8> buf[3][MAX_WIDTH])
{
    #pragma HLS INLINE
    buf[row_idx][col] = pix.data;
}

static void window_channel(
    int row, int col,
    int height, int width,
    ap_uint<8> buf[3][MAX_WIDTH],
    ap_uint<8> win[3][3])
{
    #pragma HLS INLINE

    // create a 3x3 window centered at current pixel location
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int r = row + i - 1;
            int c = col + j - 1;
            r = (r < 0) ? 0 : (r >= height ? height - 1 : r);
            c = (c < 0) ? 0 : (c >= width  ? width  - 1 : c);
            win[i][j] = buf[r % 3][c];
        }
}

static void apply_channel(
    ap_uint<8> win[3][3],
    int kernel_factor,
    int kernel[3][3],
    AXI_PIXEL &pix_out)
{
    #pragma HLS INLINE
    int acc = 0;

    // 3×3 convolution
    for (int i = 0; i < 3; ++i)
        #pragma HLS UNROLL
        for (int j = 0; j < 3; ++j)
            #pragma HLS UNROLL
            acc += int(win[i][j]) * kernel[i][j];

    // Normalize + clip values
    acc /= kernel_factor;
    if (acc <   0) acc = 0;
    if (acc > 255) acc = 255;

    pix_out.data = ap_uint<8>(acc);
}


void filter_kernel(
    int image_width,
    int image_height,
    int kernel_factor,
    int kernel[3][3],

    hls::stream<AXI_PIXEL> &input_stream,
    hls::stream<AXI_PIXEL> &output_stream,

    hls::stream<AXI_PIXEL> &input_stream2,
    hls::stream<AXI_PIXEL> &output_stream2,

    hls::stream<AXI_PIXEL> &input_stream3,
    hls::stream<AXI_PIXEL> &output_stream3)
{

    // AXI-Lite slave interface, lower bandwidth interface for control signals
    // `bundle=control` creates one shared AXI-Lite interface for all control parameters, in hardware this becomes a set of memory-mapped registers accessible from the PS

    #pragma HLS INTERFACE s_axilite port=image_width    bundle=control
    #pragma HLS INTERFACE s_axilite port=image_height   bundle=control
    #pragma HLS INTERFACE s_axilite port=kernel_factor  bundle=control
    #pragma HLS INTERFACE s_axilite port=kernel         bundle=control
    #pragma HLS INTERFACE s_axilite port=return         bundle=control

    // #pragma HLS INTERFACE s_axilite port=channels bundle=control
    // `port=return` creates 4 hardware signals
    // `ap_start` software set to '1' to begin execution
    // `ap_done` goes to '1' when processing is complete
    // `ap_idle` `1` when the IP block is not running
    // `ap_ready` `1` when the IP block is accepting new signals

    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE axis port=input_stream2
    #pragma HLS INTERFACE axis port=output_stream2
    #pragma HLS INTERFACE axis port=input_stream3
    #pragma HLS INTERFACE axis port=output_stream3

    // line buffers: one per color channel, 3 rows deep
    ap_uint<8> bufR[3][MAX_WIDTH];
    ap_uint<8> bufG[3][MAX_WIDTH];
    ap_uint<8> bufB[3][MAX_WIDTH];

    #pragma HLS RESOURCE variable=bufR core=RAM_S2P_BRAM
    #pragma HLS RESOURCE variable=bufG core=RAM_S2P_BRAM
    #pragma HLS RESOURCE variable=bufB core=RAM_S2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=bufR complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bufG complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bufB complete dim=1

    ap_uint<8> winR[3][3], winG[3][3], winB[3][3];
    #pragma HLS ARRAY_PARTITION variable=winR complete dim=0
    #pragma HLS ARRAY_PARTITION variable=winG complete dim=0
    #pragma HLS ARRAY_PARTITION variable=winB complete dim=0


    for (int row = 0; row < image_height; ++row)
        for (int col = 0; col < image_width; ++col) {
            #pragma HLS PIPELINE II=1
            int idx = row % 3;

            // red channel
            AXI_PIXEL pr;  input_stream  >> pr;
            store_channel (pr, idx, col, bufR);
            window_channel(row, col, image_height, image_width, bufR, winR);
            apply_channel (winR, kernel_factor, kernel, pr);
            pr.last = (row == image_height - 1 && col == image_width - 1);
            output_stream << pr;

            // green channel
            AXI_PIXEL pg;  input_stream2 >> pg;
            store_channel (pg, idx, col, bufG);
            window_channel(row, col, image_height, image_width, bufG, winG);
            apply_channel (winG, kernel_factor, kernel, pg);
            pg.last = (row == image_height - 1 && col == image_width - 1);
            output_stream2 << pg;

            // blue channel
            AXI_PIXEL pb;  input_stream3 >> pb;
            store_channel (pb, idx, col, bufB);
            window_channel(row, col, image_height, image_width, bufB, winB);
            apply_channel (winB, kernel_factor, kernel, pb);
            pb.last = (row == image_height - 1 && col == image_width - 1);
            output_stream3 << pb;
        }
}
