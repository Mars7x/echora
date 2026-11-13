# Shader translator plugins

Echora turns the game's GPU programs into SPIR-V with its own translator,
`echora-shader` (see [shader-translator.md](shader-translator.md)), which
handles every program the game has used so far. An external translator
can still be loaded at run time as an optional plugin, to compare with or
to fall back on. It is used only when selected with
`ECHORA_SHADER_TRANSLATOR=plugin` (or `compare`).
No plugin is part of this repository; one is built and installed
separately, under its own license.

## Loading

The runtime loads `ECHORA_SHADER_PLUGIN` (a shared library path), else
`echora/shader-plugin.so` in the cache directory (`$XDG_CACHE_HOME`, or
`~/.cache`). Plugins load on Linux only. `echora run` sets
`EDEN_NX2_TENSOR=1`, which a plugin may read to expand matrix multiplies. A plugin built for a different
`echora_shader_abi::ABI_VERSION`, or whose input and output structs differ
in size from Echora's, is not used.

`ECHORA_SHADER_TRANSLATOR` chooses among the translators:

| Value | Programs translated by |
|---|---|
| `rust` (the default) | the Rust translator where it can, an installed plugin otherwise |
| `plugin` | the plugin |
| `compare` | the plugin; how the Rust translator differs is reported |

`shader-check --compare stage-*.code` makes the same comparison offline
over captured programs.

## Interface

A plugin exports five C functions:

| Symbol | Signature | Meaning |
|---|---|---|
| `echora_plugin_abi_version` | `uint32_t (void)` | The `ABI_VERSION` its programs are made for |
| `echora_plugin_input_size` | `size_t (void)` | `sizeof` its input struct |
| `echora_plugin_output_size` | `size_t (void)` | `sizeof` its output struct |
| `echora_plugin_translate` | `int (const Input*, Output*)` | Translate one program: nonzero on success; on failure, a message in `Output.error` |
| `echora_plugin_free` | `void (Output*)` | Free what a successful translate allocated (`Output.spirv`) |

The structs are C layouts, in this field order (`crates/echora-gpu/src/shader.rs`
mirrors them):

- **Input**: `program` (bytes as NVN stores them at the stage's GPU address;
  graphics programs start with their 128-byte header) and `length`;
  `workgroup_size[3]`, `shared_memory_size`, `local_memory_size` (compute);
  `binding_base`; `has_previous_stores` and `previous_stores[8]` (the
  previous stage's stored attributes); `generic_input_types[32]`;
  `resolution_scale_numerator` and `_denominator`; `texture_scaling_base`,
  `image_scaling_base`; `mesh_stage` (0 none, 1 task, 2 mesh),
  `mesh_registers[3]`, `has_mesh_primitive_header` and
  `mesh_primitive_header[0x50]`; `callbacks`; `native_pointers` (nonzero:
  storage buffer pointers are raw 8-byte arguments).
- **Callbacks**: `user` and four functions the plugin may call during
  translation: `read_cbuf(user, index, offset)` (a constant buffer word),
  `texture_type(user, handle)`, `is_storage_pointer(user, index, offset)`
  and `is_indirect_storage_pointer(user, index, offset, indirect)`. A null
  function means the default answer (0, a 2D texture, yes, yes).
- **Output**: `spirv` and `words`; `stage`; `descriptors[64]`
  (`echora_shader_abi::Descriptor`) and `descriptor_count`;
  `loaded_generics`, `stored_generics`; `stores[8]`; `frag_colors`;
  `cbuf_used_sizes[18]`; `error[256]`.

Descriptors are numbered in set 0 from `binding_base` in the order listed.
Programs read the push-constant block of `echora_shader_abi::PushConstants`
(`docs/shader-translator.md`).
