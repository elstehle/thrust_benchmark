#include <nvbench/detail/throw.cuh>
#include <nvbench/nvbench.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <cub/device/device_memcpy.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <tbm/range_generator.cuh>

#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

template <typename ItT, typename T>
void __global__ prnt(ItT data, T num_items)
{
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < num_items; i++)
    {
      auto x = data[i];
      printf("%d: %d\n", i, x);
    }
  }
}

/**
 * @brief Enum class with options for generating the buffer order within memory
 */
enum class buffer_order
{
  // Buffers are randomly shuffled within memory
  RANDOM,

  // Buffer N+1 resides next to buffer N
  CONSECUTIVE
};

/**
 * @brief Function object class template that takes an offset and returns an
 * iterator at the given offset relative to a fixed base iterator.
 *
 * @tparam IteratorT The random-access iterator type to be returned
 */
template <typename IteratorT>
struct offset_to_ptr_op
{
  template <typename T>
  __host__ __device__ __forceinline__ IteratorT operator()(T offset) const
  {
    return base_it + offset;
  }
  IteratorT base_it;
};

/**
 * @brief Used for generating a shuffled but cohesive sequence of output-buffer
 * offsets for the sequence of input-buffers.
 */
template <typename BufferOffsetT,
          typename ByteOffsetT,
          typename BufferSizeItT,
          typename ByteOffsetItT>
void get_shuffled_buffer_offsets(BufferSizeItT buffer_sizes,
                                 BufferOffsetT num_buffers,
                                 ByteOffsetItT shuffled_buffer_offsets,
                                 const unsigned int seed = 320981U)
{
  num_buffers = 100;

  // We're remapping the i-th buffer to pmt_idxs[i]
  thrust::device_vector<BufferOffsetT> pmt_idxs(num_buffers);

  thrust::default_random_engine g{seed};
  auto seq_it = thrust::make_counting_iterator(BufferOffsetT{0});
  thrust::shuffle_copy(seq_it, seq_it + num_buffers, std::begin(pmt_idxs), g);

  // std::cout << " -- Buffer sizes --\n";
  // prnt<<<1, 1>>>(buffer_sizes, num_buffers);

  // std::cout << " -- Permutation indexes --\n";
  // for (int i = 0; i < num_buffers; i++)
  //   std::cout << i << ": " << pmt_idxs[i] << "\n";
  // exit(-1);

  // Compute the offsets using the new mapping
  auto permuted_sizes_it =
    thrust::make_permutation_iterator(buffer_sizes, std::begin(pmt_idxs));
  thrust::device_vector<ByteOffsetT> permuted_offsets(num_buffers);
  thrust::exclusive_scan(permuted_sizes_it,
                         permuted_sizes_it + num_buffers,
                         std::begin(permuted_offsets),
                         static_cast<ByteOffsetT>(0));

  // for (auto permuted_buffer_idx : pmt_idxs)
  // {
  //   permuted_offsets.emplace_back(running_offset);
  //   running_offset += buffer_sizes[permuted_buffer_idx];
  // }

  // Generate the scatter indexes that identify where each buffer was mapped to
  thrust::device_vector<BufferOffsetT> gather_map(num_buffers);
  thrust::scatter(seq_it,
                  seq_it + num_buffers,
                  std::begin(pmt_idxs),
                  std::begin(gather_map));
  // thrust::sequence(permuted_scatter_idxs, permuted_scatter_idxs +
  // num_buffers, BufferOffsetT{0}); for (BufferOffsetT i = 0; i < num_buffers;
  // i++)
  // {
  //   scatter_idxs[pmt_idxs[i]] = i;
  // }

  thrust::device_vector<ByteOffsetT> new_offsets(num_buffers);
  thrust::gather(std::begin(gather_map),
                 std::end(gather_map),
                 std::begin(permuted_offsets),
                 shuffled_buffer_offsets);
  //  std::begin(new_offsets));
  // for (BufferOffsetT i = 0; i < num_buffers; i++)
  // {
  //   new_offsets[i] = permuted_offsets[scatter_idxs[i]];
  // }
}

struct offset_buffers_and_resize_op
{
  std::size_t min_buffer_size{};
  std::size_t max_buffer_size{};
  std::size_t atomic_type_size{};

  template <typename T>
  __device__ __forceinline__ T operator()(const T &val)
  {
    auto rnd_buffer_size = std::abs(val);
    rnd_buffer_size = rnd_buffer_size % (max_buffer_size - min_buffer_size);
    rnd_buffer_size = min_buffer_size + rnd_buffer_size;
    return CUB_ROUND_DOWN_NEAREST(rnd_buffer_size,
                                  static_cast<T>(atomic_type_size));
  }
};

template <typename AtomicT, buffer_order buffer_order>
static void basic(nvbench::state &state,
                  nvbench::type_list<AtomicT, nvbench::enum_type<buffer_order>>)
{
  // Type alias
  using SrcPtrT       = uint8_t *;
  using BufferOffsetT = int32_t;
  using BufferSizeT   = int32_t;
  using ByteOffsetT   = int32_t;

  constexpr auto input_gen  = buffer_order;
  constexpr auto output_gen = buffer_order;

  const auto target_copy_size =
    static_cast<std::size_t>(state.get_int64("Elements"));

  // Make sure buffer ranges are an integer multiple of AtomicT
  const auto min_buffer_size = CUB_ROUND_UP_NEAREST(
    static_cast<std::size_t>(state.get_int64("Min. buffer size")),
    sizeof(AtomicT));
  const auto max_buffer_size = CUB_ROUND_UP_NEAREST(
    static_cast<std::size_t>(state.get_int64("Max. buffer size")),
    sizeof(AtomicT));

  // Skip benchmarks where min. buffer size exceeds max. buffer size
  if (min_buffer_size > max_buffer_size)
  {
    state.skip("Skipping benchmark, as min. buffer size exceeds max. buffer "
               "size.");
    return;
  }

  // Compute number of buffers to generate
  double average_buffer_size = (min_buffer_size + max_buffer_size) / 2.0;
  const auto num_buffers =
    static_cast<std::size_t>(target_copy_size / average_buffer_size);

  // Buffer segment data (their offsets and sizes)
  thrust::device_vector<ByteOffsetT> h_buffer_src_offsets(num_buffers);
  thrust::device_vector<ByteOffsetT> h_buffer_dst_offsets(num_buffers);

  // Generate the buffer sizes
  auto rnd_buffer_sizes =
    tbm::make_range_generator<BufferSizeT,
                              tbm::iterator_style::pointer,
                              tbm::data_pattern::random>(num_buffers);

  // Offset random buffer sizes by the minimum buffer size and make sure buffer
  // sizes are a multiple of the most granular unit (one AtomicT) being copied
  // (round down)
  // prnt<<<1, 1>>>(rnd_buffer_sizes.cbegin(), 100);
  thrust::device_vector<BufferSizeT> buffer_sizes(num_buffers);
  thrust::transform(std::cbegin(rnd_buffer_sizes),
                    std::cend(rnd_buffer_sizes),
                    std::begin(buffer_sizes),
                    offset_buffers_and_resize_op{min_buffer_size,
                                                 max_buffer_size,
                                                 sizeof(AtomicT)});
  // prnt<<<1, 1>>>(thrust::raw_pointer_cast(buffer_sizes.data()), 100);

  // Make sure buffer sizes are a multiple of the most granular unit (one
  // AtomicT) being copied (round down)
  //   for (BufferOffsetT i = 0; i < num_buffers; i++)
  // {
  //   h_buffer_sizes[i] = (h_buffer_sizes[i] / sizeof(AtomicT)) *
  //   sizeof(AtomicT);
  // }

  // For consecutive input buffer order generate the buffer offsets
  if (input_gen == buffer_order::CONSECUTIVE)
  {
    thrust::exclusive_scan(std::begin(buffer_sizes),
                           std::end(buffer_sizes),
                           std::begin(h_buffer_src_offsets),
                           0);
  }
  // For consecutive output buffer order generate the buffer offsets
  if (output_gen == buffer_order::CONSECUTIVE)
  {
    thrust::exclusive_scan(std::begin(buffer_sizes),
                           std::end(buffer_sizes),
                           std::begin(h_buffer_dst_offsets),
                           0);
  }

  // Compute the total bytes to be copied
  ByteOffsetT num_total_bytes = thrust::reduce(std::begin(buffer_sizes),
                                               std::end(buffer_sizes));

  // Shuffle input buffer source-offsets
  unsigned int shuffle_seed = 320981U;
  if (input_gen == buffer_order::RANDOM)
  {
    get_shuffled_buffer_offsets<BufferOffsetT, ByteOffsetT>(
      buffer_sizes.cbegin(),
      static_cast<BufferOffsetT>(buffer_sizes.size()),
      h_buffer_src_offsets.data(),
      shuffle_seed);
    shuffle_seed += 42;
  }

  // Shuffle input buffer source-offsets
  if (output_gen == buffer_order::RANDOM)
  {
    get_shuffled_buffer_offsets<BufferOffsetT, ByteOffsetT>(
      buffer_sizes.cbegin(),
      static_cast<BufferOffsetT>(buffer_sizes.size()),
      h_buffer_dst_offsets.data(),
      shuffle_seed);
  }

  // Get temporary storage requirements
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceMemcpy::Batched(nullptr,
                                          temp_storage_bytes,
                                          static_cast<SrcPtrT *>(nullptr),
                                          static_cast<SrcPtrT *>(nullptr),
                                          static_cast<BufferSizeT *>(nullptr),
                                          num_buffers));

  // Compute total device memory requirements
  std::size_t total_required_mem = num_total_bytes +                     //
                                   num_total_bytes +                     //
                                   (num_buffers * sizeof(ByteOffsetT)) + //
                                   (num_buffers * sizeof(ByteOffsetT)) + //
                                   (num_buffers * sizeof(BufferSizeT)) + //
                                   temp_storage_bytes;                   //

  // Get available device memory
  std::size_t available_device_mem =
    state.get_device().has_value()
      ? state.get_device().value().get_global_memory_usage().bytes_free
      : 0;

  // Skip benchmark there's insufficient device memory available
  if (available_device_mem < total_required_mem)
  {
    state.skip("Skipping benchmark due to insufficient device memory");
    return;
  }

  thrust::device_vector<uint8_t> d_temp_storage(temp_storage_bytes);

  // Add benchmark reads
  state.add_element_count(num_total_bytes);
  state.add_global_memory_reads<char>(num_total_bytes, "data");
  state.add_global_memory_reads<ByteOffsetT>(num_buffers, "buffer src offsets");
  state.add_global_memory_reads<ByteOffsetT>(num_buffers, "buffer dst offsets");
  state.add_global_memory_reads<BufferSizeT>(num_buffers, "buffer sizes");

  // Add benchmark writes
  state.add_global_memory_writes<char>(num_total_bytes, "data");

  // Prepare random data segment (which serves for the buffer sources)
  thrust::device_vector<uint8_t> d_in_buffer(num_total_bytes);
  thrust::device_vector<uint8_t> d_out_buffer(num_total_bytes);

  // Populate the data source buffer
  thrust::fill(std::begin(d_in_buffer),
               std::end(d_in_buffer),
               std::numeric_limits<uint8_t>::max());

  // Raw pointers into the source and destination buffer
  auto d_in  = thrust::raw_pointer_cast(d_in_buffer.data());
  auto d_out = thrust::raw_pointer_cast(d_out_buffer.data());

  // Prepare device-side data
  thrust::device_vector<ByteOffsetT> d_buffer_src_offsets =
    h_buffer_src_offsets;
  thrust::device_vector<ByteOffsetT> d_buffer_dst_offsets =
    h_buffer_dst_offsets;
  thrust::device_vector<BufferSizeT> d_buffer_sizes = buffer_sizes;

  // Prepare d_buffer_srcs
  offset_to_ptr_op<SrcPtrT> src_transform_op{static_cast<SrcPtrT>(d_in)};
  cub::TransformInputIterator<SrcPtrT, offset_to_ptr_op<SrcPtrT>, ByteOffsetT *>
    d_buffer_srcs(thrust::raw_pointer_cast(d_buffer_src_offsets.data()),
                  src_transform_op);

  // Prepare d_buffer_dsts
  offset_to_ptr_op<SrcPtrT> dst_transform_op{static_cast<SrcPtrT>(d_out)};
  cub::TransformInputIterator<SrcPtrT, offset_to_ptr_op<SrcPtrT>, ByteOffsetT *>
    d_buffer_dsts(thrust::raw_pointer_cast(d_buffer_dst_offsets.data()),
                  dst_transform_op);

  state.exec([&](nvbench::launch &launch) {
    std::size_t temp_size = d_temp_storage.size(); // need an lvalue
    cub::DeviceMemcpy::Batched(thrust::raw_pointer_cast(d_temp_storage.data()),
                               temp_size,
                               d_buffer_srcs,
                               d_buffer_dsts,
                               thrust::raw_pointer_cast(d_buffer_sizes.data()),
                               num_buffers,
                               launch.get_stream());
  });
}

// Column names for type axes:
inline std::vector<std::string> type_axis_names()
{
  return {"AtomicT", "Buffer Order"};
}

// Benchmark for unaligned buffers and buffers aligned to four bytes
using atomic_type = nvbench::type_list<nvbench::uint8_t, nvbench::uint32_t>;

using buffer_orders =
  nvbench::enum_type_list<buffer_order::RANDOM, buffer_order::CONSECUTIVE>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  buffer_order,
  [](buffer_order data_gen_mode) {
    switch (data_gen_mode)
    {
      case buffer_order::RANDOM:
        return "Random";
      case buffer_order::CONSECUTIVE:
        return "Consecutive";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  },
  [](buffer_order data_gen_mode) {
    switch (data_gen_mode)
    {
      case buffer_order::RANDOM:
        return "Buffers are randomly shuffled within memory";
      case buffer_order::CONSECUTIVE:
        return "Consecutive buffers reside cohesively in memory";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  })

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(atomic_type, buffer_orders))
  .set_name("cub::DeviceMemcpy::Batched")
  .set_type_axes_names(type_axis_names())
  .add_int64_axis("Min. buffer size", {1, 64 * 1024})
  .add_int64_axis("Max. buffer size", {8, 64, 256, 1024, 64 * 1024})
  .add_int64_power_of_two_axis("Elements", nvbench::range(25, 29, 2));
