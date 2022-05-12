#include "block_exchange.cuh"

struct warp_striped_to_blocked
{
  template <typename BlockExchange, typename T, int ItemsPerThread>
  __device__ int operator()(BlockExchange &block_exchange,
                            T (&thread_data)[ItemsPerThread])
  {
    block_exchange.WarpStripedToBlocked(thread_data, thread_data);
    return 0; // All items have defined values
  }
};

using op = nvbench::type_list<warp_striped_to_blocked>;

NVBENCH_BENCH_TYPES(bench,
                    NVBENCH_TYPE_AXES(types,
                                      op,
                                      threads_in_block,
                                      items_per_thread,
                                      compute_modes))
  .set_name("cub::BlockExchange::WarpStripedToBlocked")
  .set_type_axes_names(block_exchange_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 22, 2));
