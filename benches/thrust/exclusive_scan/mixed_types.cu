#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

using value_types = nvbench::type_list<nvbench::int32_t, nvbench::float32_t>;

template <typename InputType, typename OutputType, typename InitialValueType>
void mixed_types(nvbench::state &state,
                 nvbench::type_list<InputType, OutputType, InitialValueType>)
{
  const auto size = state.get_int64("Size");

  thrust::device_vector<InputType> input(size);
  thrust::device_vector<OutputType> output(size);

  thrust::sequence(input.begin(), input.end());

  const auto input_bytes  = size * sizeof(InputType);
  const auto output_bytes = size * sizeof(OutputType);
  state.set_global_bytes_accessed_per_launch(input_bytes + output_bytes);
  state.set_items_processed_per_launch(size);

  nvbench::exec(state, [&input, &output](nvbench::launch &launch) {
    thrust::exclusive_scan(thrust::device.on(launch.get_stream()),
                           input.cbegin(),
                           input.cend(),
                           output.begin());
  });
}
NVBENCH_CREATE_TEMPLATE(mixed_types,
                        NVBENCH_TYPE_AXES(value_types, value_types, value_types))
  .set_name("thrust::exclusive_scan (mixed types)")
  .set_type_axes_names({"In", "Out", "Init"})
  .add_int64_power_of_two_axis("Size", nvbench::range(20, 28, 4));

NVBENCH_MAIN;