#pragma once

#include <nvbench/types.cuh>

#include <string>
#include <unordered_map>
#include <variant>

namespace nvbench
{

struct state
{
  // move-only
  state(const state &) = delete;
  state(state &&) = default;
  state &operator=(const state &) = delete;
  state &operator=(state &&) = default;

  [[nodiscard]] nvbench::int64_t get_int64(const std::string &axis_name) const;

  [[nodiscard]] nvbench::float64_t
  get_float64(const std::string &axis_name) const;

  [[nodiscard]] const std::string &
  get_string(const std::string &axis_name) const;

protected:
  state() = default;

  using param_type =
    std::variant<nvbench::int64_t, nvbench::float64_t, std::string>;
  using params_type = std::unordered_map<std::string, param_type>;

  explicit state(params_type params)
      : m_params{std::move(params)}
  {}

  void set_param(std::string axis_name, nvbench::int64_t value);
  void set_param(std::string axis_name, nvbench::float64_t value);
  void set_param(std::string axis_name, std::string value);

  params_type m_params;
};

} // namespace nvbench