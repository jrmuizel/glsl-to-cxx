extern crate glsl;

use glsl::parser::Parse;
use glsl::syntax::{TranslationUnit, UnaryOp};

mod hir;

use hir::Type;
use std::io::Read;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::mem;

#[derive(PartialEq, Eq)]
enum ShaderKind {
  Fragment,
  Vertex
}

fn build_uniform_indices(indices: &mut BTreeMap<String, i32>, state: &hir::State) {
  for u in state.used_globals.borrow().iter() {
    let sym = state.sym(*u);
    if let hir::SymDecl::Global(hir::StorageClass::Uniform, ..) = &sym.decl {
      let next_index = indices.len() as i32 + 1;
      indices.entry(sym.name.clone()).or_insert(next_index);
    }
  }
}

fn main() {
  let vertex_file = std::env::args().nth(1).unwrap();
  let vs_name = vertex_file.split(".").next().unwrap().to_owned();

  let frag_file = std::env::args().nth(2).unwrap();
  let fs_name = frag_file.split(".").next().unwrap().to_owned();

  let (vs_state, vs_hir, vs_is_frag) = parse_shader(vertex_file);
  let (fs_state, fs_hir, fs_is_frag) = parse_shader(frag_file);

  // we use a BTree so that iteration is stable
  let mut uniform_indices = BTreeMap::new();
  build_uniform_indices(&mut uniform_indices, &vs_state);
  build_uniform_indices(&mut uniform_indices, &fs_state);

  assert_eq!(fs_name, vs_name);

  translate_shader(vs_name, vs_state, vs_hir, vs_is_frag, &uniform_indices);
  translate_shader(fs_name, fs_state, fs_hir, fs_is_frag, &uniform_indices);
}

fn parse_shader(file: String) -> (hir::State, hir::TranslationUnit, bool) {
  use std::io::Write;

  let mut contents = String::new();
  let is_frag = file.contains("frag");
  std::fs::File::open(&file).unwrap().read_to_string(&mut contents).unwrap();
  let r = TranslationUnit::parse(contents);

  //println!("{:#?}", r);
  let mut ast_glsl = String::new();
  let r = r.unwrap();
  glsl::transpiler::glsl::show_translation_unit(&mut ast_glsl, &r);
  let mut fast = std::fs::File::create("ast").unwrap();
  fast.write(ast_glsl.as_bytes());

  let mut state = hir::State::new();
  let hir = hir::ast_to_hir(&mut state, &r);
  (state, hir, is_frag)
}

fn translate_shader(name: String, mut state: hir::State, hir: hir::TranslationUnit, is_frag: bool, uniform_indices: &BTreeMap<String, i32>) {
  use std::io::Write;

  //println!("{:#?}", state);

  hir::infer_run_class(&mut state, &hir);

  let mut uniforms = Vec::new();
  let mut inputs = Vec::new();
  let mut outputs = Vec::new();

  for i in &hir {
    match i {
      hir::ExternalDeclaration::Declaration(hir::Declaration::InitDeclaratorList(ref d))  => {
        match &state.sym(d.head.name).decl {
          hir::SymDecl::Global(storage, ..) if state.used_globals.borrow().contains(&d.head.name) => {
            match storage {
              hir::StorageClass::Uniform => {
                uniforms.push(d.head.name);
              }
              hir::StorageClass::In => {
                inputs.push(d.head.name);
              }
              hir::StorageClass::Out => {
                outputs.push(d.head.name);
              }
              _ => {}
            }
          }
          _ => {}
        }
      }
      _ => {}
    }
  }

  //println!("{:#?}", hir);

  let mut state = OutputState {
    hir: state,
    output: String::new(),
    buffer: RefCell::new(String::new()),
    indent: 0,
    should_indent: false,
    output_cxx: false,
    mask: None,
    cond_index: 0,
    return_type: None,
    return_declared: false,
    return_vector: false,
    is_scalar: Cell::new(false),
    is_lval: Cell::new(false),
    kind: if is_frag { ShaderKind::Fragment } else { ShaderKind::Vertex },
    functions: HashMap::new(),
    deps: RefCell::new(Vec::new()),
    vector_mask: 0,
    uses_discard: false,
  };

  show_translation_unit(&mut state, &hir);
  let output_glsl = state.finish_output();

  state.should_indent = true;
  state.output_cxx = true;


  write!(state, "/* inputs\n");
  for i in &inputs {
    show_variable(&mut state, &i);
    write!(state, "\n");
  }
  write!(state, "*/\n");

  write!(state, "/* outputs\n");
  for i in &outputs {
    show_variable(&mut state, &i);
    write!(state, "\n");
  }
  write!(state, "*/\n");

  if state.output_cxx {
    let part_name = name.to_owned() +
        match state.kind {
          ShaderKind::Vertex => "_vert",
          ShaderKind::Fragment => "_frag",
        };
    let shader_impl = match state.kind {
        ShaderKind::Vertex => "VertexShaderImpl",
        ShaderKind::Fragment => "FragmentShaderImpl",
    };
    write!(state, "struct {} : {} {{\n", part_name, shader_impl);
    write_set_uniform_1i(&mut state, &uniforms, uniform_indices);
    write_set_uniform_4fv(&mut state, &uniforms, uniform_indices);
    write_set_uniform_matrix4fv(&mut state, &uniforms, uniform_indices);
    if state.kind == ShaderKind::Vertex {
      write_get_uniform_index(&mut state, &uniform_indices);
      write_bind_attrib_location(&mut state, &inputs);
      write_load_attribs(&mut state, &inputs);
      write_store_outputs(&mut state, &outputs);
    } else {
      write_read_inputs(&mut state, &inputs, name.to_owned() + "_vert");
    }
    write_bind_textures(&mut state, &uniforms);

    show_translation_unit(&mut state, &hir);

    write_abi(&mut state, &part_name);
    write!(state, "}};");
  } else {
    show_translation_unit(&mut state, &hir);
  }
  let output_cxx = state.finish_output();


  let mut hir = std::fs::File::create("hir").unwrap();
  hir.write(output_glsl.as_bytes());


  println!("{}", output_cxx);
}

fn write_get_uniform_index(state: &mut OutputState, uniform_indices: &BTreeMap<String, i32>) {
  write!(state, "int get_uniform_location(const char *name) {{\n");
  for (uniform_name, index) in uniform_indices.iter() {
    write!(state, "if (strcmp(\"{}\", name) == 0) {{ return {}; }}\n", uniform_name, index);
  }
  write!(state, "return -1;\n");
  write!(state, "}}\n");
}

fn float4_compatible(ty: hir::TypeKind) -> bool {
  match ty {
    _ => false
  }
}

fn matrix4_compatible(ty: hir::TypeKind) -> bool {
  match ty {
    hir::TypeKind::Mat4 => true,
    _ => false
  }
}

fn write_bind_textures(state: &mut OutputState, uniforms: &[hir::SymRef]) {
  for i in uniforms {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, _) => {
        let name = sym.name.as_str();
        match ty.kind {
          hir::TypeKind::Sampler2D | hir::TypeKind::ISampler2D | hir::TypeKind::Sampler2DArray => {
            show_type(state, &ty);
            write!(state, "_impl {}_impl;\n", name);
            write!(state, "int {}_slot;\n", name);
          }
          _ => {}
        };
      }
      _ => panic!()
    }
  }

  write!(state, "void bind_textures() {{\n");
  for i in uniforms {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, _) => {
        let name = sym.name.as_str();
        match ty.kind {
          hir::TypeKind::Sampler2D => write!(state, "{} = lookup_sampler(&{}_impl, {}_slot);\n", name, name, name),
          hir::TypeKind::ISampler2D => write!(state, "{} = lookup_isampler(&{}_impl, {}_slot);\n", name, name, name),
          hir::TypeKind::Sampler2DArray => write!(state, "{} = lookup_sampler_array(&{}_impl, {}_slot);\n", name, name, name),
          _ => {}
        };
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");
}

fn write_set_uniform_1i(state: &mut OutputState, uniforms: &[hir::SymRef], uniform_indices: &BTreeMap<String, i32>) {
  write!(state, "void set_uniform_1i(int index, int value) {{\n");
  for i in uniforms {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, _) => {
        let name = sym.name.as_str();
        let index = uniform_indices.get(name).unwrap();
        write!(state, "if (index == {}) {{\n", index);
        match ty.kind {
          hir::TypeKind::Int => write!(state, "{} = {}(value);\n", name, scalar_type_name(state, ty)),
          hir::TypeKind::Sampler2D |
          hir::TypeKind::ISampler2D |
          hir::TypeKind::Sampler2DArray => write!(state, "{}_slot = value;\n", name),
          _ => write!(state, "assert(0); // {}\n", name),
        };
        write!(state, "}}\n");
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");
}

fn write_set_uniform_4fv(state: &mut OutputState, uniforms: &[hir::SymRef], uniform_indices: &BTreeMap<String, i32>) {
  write!(state, "void set_uniform_4fv(int index, const float *value) {{\n");
  for i in uniforms {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, _) => {
        let name = sym.name.as_str();
        let index = uniform_indices.get(name).unwrap();
        write!(state, "if (index == {}) {{\n", index);
        if float4_compatible(ty.kind.clone()) {
          write!(state, "{} = {}(value);\n", name, scalar_type_name(state, ty));
        } else {
          write!(state, "assert(0); // {}\n", name);
        }
        write!(state, "}}\n");
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");
}

fn write_set_uniform_matrix4fv(state: &mut OutputState, uniforms: &[hir::SymRef], uniform_indices: &BTreeMap<String, i32>) {
  write!(state, "void set_uniform_matrix4fv(int index, const float *value) {{\n");
  for i in uniforms {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, _) => {
        let name = sym.name.as_str();
        let index = uniform_indices.get(name).unwrap();

        write!(state, "if (index == {}) {{\n", index);
        if matrix4_compatible(ty.kind.clone()) {
          write!(state, "{} = mat4_scalar::load_from_ptr(value);\n", name);
        } else {
          write!(state, "assert(0); // {}\n", name);
        }
        write!(state, "}}\n");
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");
}

fn write_bind_attrib_location(state: &mut OutputState, attribs: &[hir::SymRef]) {
  let mut index = 1;
  for i in attribs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, _) => {
        write!(state, "static inline int {}_location_index;\n", sym.name.as_str());
        index += 1;
      }
      _ => panic!()
    }
  }

  write!(state, "void bind_attrib_location(const char *name, int index) {{\n");
  for i in attribs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, _) => {
        write!(state, "if (strcmp(\"{}\", name) == 0) {{ {}_location_index = index; }}\n", sym.name.as_str(), sym.name.as_str());
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");
}

fn scalar_type_name(state: &OutputState, ty: &Type) -> String {
  let kind_name = if let Some(name) = ty.kind.cxx_primitive_scalar_type_name() {
    name.into()
  } else {
    let buffer = state.push_buffer();
    show_type(state, ty);
    state.pop_buffer(buffer) + "_scalar"
  };
  if let Some(ref array) = ty.array_sizes {
    let size = match &array.sizes[..] {
      [size] => size,
      _ => panic!()
    };
    let buffer = state.push_buffer();
    show_hir_expr(state, size);
    let size_string = state.pop_buffer(buffer);

    format!("std::array<{}, {}>", kind_name, size_string)
  } else {
    kind_name
  }
}

fn type_name(state: &OutputState, ty: &Type) -> String {
  let buffer = state.push_buffer();
  show_type(state, ty);
  state.pop_buffer(buffer)
}

fn write_load_attribs(state: &mut OutputState, attribs: &[hir::SymRef]) {
  write!(state, "void load_attribs(VertexAttrib *attribs, unsigned short *indices, int start, int instance, int count) {{\n");
  for i in attribs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, interpolation, ty, run_class) => {
        let name = sym.name.as_str();
        write!(state, "  load_attrib({}, attribs[{}_location_index], indices, start, instance, count);\n", name, name);
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");
}

fn write_store_outputs(state: &mut OutputState, outputs: &[hir::SymRef]) {
  let is_scalar = state.is_scalar.replace(true);
  write!(state, "struct FlatOutputs {{\n");
  for i in outputs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, run_class) => {
        if *run_class == hir::RunClass::Scalar {
          show_type(state, ty);
          write!(state, " {};\n", sym.name.as_str());
        }
      }
      _ => panic!()
    }
  }
  write!(state, "}};\n");

  write!(state, "struct InterpOutputs {{\n");
  for i in outputs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, ty, run_class) => {
        if *run_class != hir::RunClass::Scalar {
          show_type(state, ty);
          write!(state, " {};\n", sym.name.as_str());
        }
      }
      _ => panic!()
    }
  }

  write!(state, "}};\n");
  state.is_scalar.set(is_scalar);

  write!(state, "ALWAYS_INLINE void store_flat_outputs(void* dest_ptr) {{\n");
  write!(state, "  auto* dest = reinterpret_cast<FlatOutputs*>(dest_ptr);\n");
  for i in outputs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, _, run_class) => {
        if *run_class == hir::RunClass::Scalar {
          let name = sym.name.as_str();
          write!(state, "  dest->{} = {};\n", name, name);
        }
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");

  write!(state, "ALWAYS_INLINE void store_interp_outputs(char* dest_ptr, size_t stride) {{\n");
  write!(state, "  for(int n = 0; n < 4; n++) {{\n");
  write!(state, "    auto* dest = reinterpret_cast<InterpOutputs*>(dest_ptr);\n");
  for i in outputs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, _, run_class) => {
        if *run_class != hir::RunClass::Scalar {
          let name = sym.name.as_str();
          write!(state, "    dest->{} = get_nth({}, n);\n", name, name);
        }
      }
      _ => panic!()
    }
  }
  write!(state, "    dest_ptr += stride;\n");
  write!(state, "  }}\n");
  write!(state, "}}\n");
}

fn write_read_inputs(state: &mut OutputState, inputs: &[hir::SymRef], vert_name: String) {
  write!(state, "typedef {}::FlatOutputs FlatInputs;\n", vert_name);
  write!(state, "typedef {}::InterpOutputs InterpInputs;\n", vert_name);

  write!(state, "void read_flat_inputs(const void* src_ptr) {{\n");
  write!(state, "  auto* src = reinterpret_cast<const FlatInputs*>(src_ptr);\n");
  for i in inputs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, _, run_class) => {
        if *run_class == hir::RunClass::Scalar {
          let name = sym.name.as_str();
          write!(state, "  {} = src->{};\n", name, name);
        }
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");

  write!(state, "void read_interp_inputs(const void* init_ptr, const void* step_ptr) {{\n");
  write!(state, "  auto* init = reinterpret_cast<const InterpInputs*>(init_ptr);\n");
  write!(state, "  auto* step = reinterpret_cast<const InterpInputs*>(step_ptr);\n");
  for i in inputs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, _, run_class) => {
        if *run_class != hir::RunClass::Scalar {
          let name = sym.name.as_str();
          write!(state, "  {0} = init_interp(init->{0}, step->{0});\n", name);
        }
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");

  write!(state, "ALWAYS_INLINE void step_interp_inputs(const void* step_ptr) {{\n");
  write!(state, "  auto* step = reinterpret_cast<const InterpInputs*>(step_ptr);\n");
  if (state.hir.used_fragcoord & 1) != 0 {
    write!(state, "  gl_FragCoord.x += 4;\n");
  }
  for i in inputs {
    let sym = state.hir.sym(*i);
    match &sym.decl {
      hir::SymDecl::Global(_, _, _, run_class) => {
        if *run_class != hir::RunClass::Scalar {
          let name = sym.name.as_str();
          write!(state, "  {} += step->{};\n", name, name);
        }
      }
      _ => panic!()
    }
  }
  write!(state, "}}\n");
}

pub struct OutputState {
  hir: hir::State,
  output: String,
  buffer: RefCell<String>,
  should_indent: bool,
  output_cxx: bool,
  indent: i32,
  mask: Option<Box<hir::Expr>>,
  cond_index: usize,
  return_type: Option<Box<hir::Type>>,
  return_declared: bool,
  return_vector: bool,
  is_scalar: Cell<bool>,
  is_lval: Cell<bool>,
  kind: ShaderKind,
  functions: HashMap<(hir::SymRef, u32), bool>,
  deps: RefCell<Vec<(hir::SymRef, u32)>>,
  vector_mask: u32,
  uses_discard: bool,
}

use std::fmt::{Arguments, Write};

impl OutputState {
  fn indent(&mut self) {
    if self.should_indent { self.indent += 1 }
  }
  fn outdent(&mut self) {
    if self.should_indent { self.indent -= 1 }
  }

  fn write(&self, s: &str) {
    self.buffer.borrow_mut().push_str(s);
  }

  fn flush_buffer(&mut self) {
    self.output.push_str(&self.buffer.borrow());
    self.buffer.borrow_mut().clear();
  }

  fn finish_output(&mut self) -> String {
    self.flush_buffer();

    let mut s = String::new();
    mem::swap(&mut self.output, &mut s);
    s
  }

  fn push_buffer(&self) -> String {
    self.buffer.replace(String::new())
  }

  fn pop_buffer(&self, s: String) -> String {
    self.buffer.replace(s)
  }

  fn write_fmt(&self, args: Arguments) {
    self.buffer.borrow_mut().write_fmt(args);
  }
}

use glsl::syntax;
use crate::hir::{SwitchStatement, SwizzleSelector, SelectionStatement, Statement};

pub fn show_identifier(state: &OutputState, i: &syntax::Identifier) {
  state.write(&i.0);
}

fn glsl_primitive_type_name_to_cxx(glsl_name: &str) -> &str {
    hir::TypeKind::from_glsl_primitive_type_name(glsl_name)
        .and_then(|kind| kind.cxx_primitive_type_name())
        .unwrap_or(glsl_name)
}

pub fn show_sym(state: &OutputState, i: &hir::SymRef) {
  let sym = state.hir.sym(*i);
  match &sym.decl {
    hir::SymDecl::NativeFunction(_, ref cxx_name) => {
      let mut name = sym.name.as_str();
      if state.output_cxx {
        name = cxx_name.unwrap_or(name);
      }
      state.write(name);
    }
    hir::SymDecl::UserFunction(..) | hir::SymDecl::Global(..) | hir::SymDecl::Local(..) | hir::SymDecl::Struct(..) => {
      let mut name = sym.name.as_str();
      // we want to replace constructor names
      if state.output_cxx {
        name = glsl_primitive_type_name_to_cxx(name);
      }
      state.write(name);
    }
    _ => panic!()
  }
}

pub fn show_variable(state: &OutputState, i: &hir::SymRef) {
  let sym = state.hir.sym(*i);
  match &sym.decl {
    hir::SymDecl::Global(_, _, ty, _) => {
      show_type(state, ty);
      state.write(" ");
      let mut name = sym.name.as_str();
      if state.output_cxx {
        name = glsl_primitive_type_name_to_cxx(name);
      }
      state.write(name);
    }
    _ => panic!()
  }
}

pub fn write_default_constructor(state: &OutputState, name: &str) {
  // write default constructor
  let _ = write!(state, "{}() = default;\n", name);
}

pub fn write_constructor(state: &OutputState, name: &str, s: &hir::StructFields) {
  let _ = write!(state, "{}(", name);
  let mut first_field = true;
  for field in &s.fields {
    if !first_field {
      state.write(", ");
    }
    show_type(state, &field.ty);
    state.write(" ");
    show_identifier_and_type(state, &field.name, &field.ty);
    first_field = false;
  }
  state.write(") : ");

  let mut first_field = true;
  for field in &s.fields {
    if !first_field {
      state.write(", ");
    }
    let _ = write!(state, "{}({})", field.name, field.name);
    first_field = false;
  }
  state.write("{}\n");
}

pub fn write_convert_constructor(state: &OutputState, name: &str, s: &hir::StructFields) {
  let _ = write!(state, "{}(", name);
  let mut first_field = true;
  for field in &s.fields {
    if !first_field {
      state.write(", ");
    }

    let is_scalar = state.is_scalar.replace(true);
    show_type(state, &field.ty);
    state.is_scalar.set(is_scalar);

    state.write(" ");

    show_identifier_and_type(state, &field.name, &field.ty);
    first_field = false;
  }
  state.write(")");

  let mut first_field = true;
  for hir::StructField { ty, name } in &s.fields {
    if ty.array_sizes.is_none() {
      if first_field {
        state.write(":");
      } else {
        state.write(",");
      }
      let _ = write!(state, "{}({})", name, name);
      first_field = false;
    }
  }
  state.write("{\n");
  for hir::StructField { ty, name } in &s.fields {
    if ty.array_sizes.is_some() {
      let _ = write!(state, "convert_array(this->{}, {});\n", name, name);
    }
  }
  state.write("}\n");

  let _ = write!(state, "{}({}_scalar s)", name, name);
  let mut first_field = true;
  for hir::StructField { ty, name } in &s.fields {
    if ty.array_sizes.is_none() {
      if first_field {
        state.write(":");
      } else {
        state.write(",");
      }
      let _ = write!(state, "{}(s.{})", name, name);
      first_field = false;
    }
  }
  state.write("{\n");
  for hir::StructField { ty, name } in &s.fields {
    if ty.array_sizes.is_some() {
      let _ = write!(state, "convert_array({}, s.{});\n", name, name);
    }
  }
  state.write("}\n");
}

pub fn write_if_then_else(state: &OutputState, name: &str, s: &hir::StructFields) {
  let _ = write!(state, "friend {} if_then_else(I32 c, {} t, {} e) {{ return {}(\n", name, name, name, name);
  let mut first_field = true;
  for field in &s.fields {
    if !first_field {
      state.write(", ");
    }
    let _ = write!(state, "if_then_else(c, t.{}, e.{})", field.name, field.name);
    first_field = false;
  }
  state.write(");\n}");
}

pub fn show_storage_class(state: &OutputState, q: &hir::StorageClass) {
  match *q {
    hir::StorageClass::None => {  }
    hir::StorageClass::Const => { state.write("const "); }
    hir::StorageClass::In => { state.write("in "); }
    hir::StorageClass::Out => { state.write("out "); }
    hir::StorageClass::Uniform => { state.write("uniform "); }
  }
}

pub fn show_sym_decl(state: &OutputState, i: &hir::SymRef) {
  let sym = state.hir.sym(*i);
  match &sym.decl {
    hir::SymDecl::Global(storage, ..) => {
      if !state.output_cxx {
        show_storage_class(state, storage)
      }
      if storage == &hir::StorageClass::Const {
        state.write("static constexpr ");
      }
      let mut name = sym.name.as_str();
      if state.output_cxx {
        name = glsl_primitive_type_name_to_cxx(name);
      }
      state.write(name);
    }
    hir::SymDecl::Local(storage, ..) => {
      if !state.output_cxx {
        show_storage_class(state, storage)
      }
      if storage == &hir::StorageClass::Const {
        state.write("const ");
      }
      let mut name = sym.name.as_str();
      if state.output_cxx {
        name = glsl_primitive_type_name_to_cxx(name);
      }
      state.write(name);
    }
    hir::SymDecl::Struct(s) => {
      let name = sym.name.as_str();

      if state.output_cxx {
        let name_scalar = format!("{}_scalar", name);
        write!(state, "struct {} {{\n", name_scalar);
        let is_scalar = state.is_scalar.replace(true);
        for field in &s.fields {
            show_struct_field(state, field);
        }
        write_default_constructor(state, &name_scalar);
        write_constructor(state, &name_scalar, s);
        state.is_scalar.set(is_scalar);
        state.write("};\n");
      }

      write!(state, "struct {} {{\n", name);
      for field in &s.fields {
        show_struct_field(state, field);
      }

      // write if_then_else
      if state.output_cxx {
        write_default_constructor(state, name);
        write_constructor(state, name, s);
        write_convert_constructor(state, name, s);
        write_if_then_else(state, name, s);
      }
      state.write("}");

    }
    _ => panic!()
  }
}

pub fn show_type_name(state: &OutputState, t: &syntax::TypeName) {
  state.write(&t.0);
}

pub fn show_type_specifier_non_array(state: &mut OutputState, t: &syntax::TypeSpecifierNonArray) {
    if let Some(kind) = hir::TypeKind::from_primitive_type_specifier(t) {
        show_type_kind(state, &kind);
    } else {
        match t {
            syntax::TypeSpecifierNonArray::Struct(ref s) => panic!(),//show_struct_non_declaration(state, s),
            syntax::TypeSpecifierNonArray::TypeName(ref tn) => show_type_name(state, tn),
            _ => unreachable!(),
        }
    }
}

pub fn show_type_kind(state: &OutputState, t: &hir::TypeKind) {
    if state.output_cxx {
        if state.is_scalar.get() {
            if let Some(name) = t.cxx_primitive_scalar_type_name() {
                state.write(name);
            } else if let Some(name) = t.cxx_primitive_type_name() {
                let mut scalar_name = String::from(name);
                scalar_name.push_str("_scalar");
                state.write(scalar_name.as_str());
            } else {
                match t {
                    hir::TypeKind::Struct(ref s) => {
                        let mut scalar_name = String::from(state.hir.sym(*s).name.as_str());
                        scalar_name.push_str("_scalar");
                        state.write(scalar_name.as_str());
                    }
                    _ => unreachable!(),
                }
            }
        } else if let Some(name) = t.cxx_primitive_type_name() {
            state.write(name);
        } else {
            match t {
                hir::TypeKind::Struct(ref s) => { state.write(state.hir.sym(*s).name.as_str()); },
                _ => unreachable!(),
            }
        }
    } else if let Some(name) = t.glsl_primitive_type_name() {
        state.write(name);
    } else {
        match t {
            hir::TypeKind::Struct(ref s) => { state.write(state.hir.sym(*s).name.as_str()); },
            _ => unreachable!(),
        }
    }
}

pub fn show_type_specifier(state: &mut OutputState, t: &syntax::TypeSpecifier) {
  show_type_specifier_non_array(state, &t.ty);

  if let Some(ref arr_spec) = t.array_specifier {
    show_array_spec(state, arr_spec);
  }
}

pub fn show_type(state: &OutputState, t: &Type) {
  if !state.output_cxx {
    if let Some(ref precision) = t.precision {
      show_precision_qualifier(state, precision);
      state.write(" ");
    }
  }

  if state.output_cxx {
    if let Some(ref array) = t.array_sizes {
      state.write("std::array<");
      show_type_kind(state, &t.kind);
      let size = match &array.sizes[..] {
        [size] => size,
        _ => panic!()
      };
      state.write(",");
      show_hir_expr(state, size);
      state.write(">");
    } else {
      show_type_kind(state, &t.kind);
    }
  } else {
    show_type_kind(state, &t.kind);
  }

  /*if let Some(ref arr_spec) = t.array_sizes {
    panic!();
  }*/
}

/*pub fn show_fully_specified_type(state: &mut OutputState, t: &FullySpecifiedType) {
  state.flat = false;
  if let Some(ref qual) = t.qualifier {
    if !state.output_cxx {
      show_type_qualifier(state, &qual);
    } else {
      state.flat = qual.qualifiers.0.iter().flat_map(|q| match q { syntax::TypeQualifierSpec::Interpolation(Flat) => Some(()), _ => None}).next().is_some();
    }
    state.write(" ");
  }

  show_type_specifier(state, &t.ty);
}*/

/*pub fn show_struct_non_declaration(state: &mut OutputState, s: &syntax::StructSpecifier) {
  state.write("struct ");

  if let Some(ref name) = s.name {
    let _ = write!(state, "{} ", name);
  }

  state.write("{\n");

  for field in &s.fields.0 {
    show_struct_field(state, field);
  }

  state.write("}");
}*/

pub fn show_struct(state: &OutputState, s: &syntax::StructSpecifier) {
  panic!();//show_struct_non_declaration(state, s);
  state.write(";\n");
}

pub fn show_struct_field(state: &OutputState, field: &hir::StructField) {
  show_type(state, &field.ty);
  state.write(" ");

  show_identifier_and_type(state, &field.name, &field.ty);

  state.write(";\n");
}

pub fn show_array_spec(state: &OutputState, a: &syntax::ArraySpecifier) {
  match *a {
    syntax::ArraySpecifier::Unsized => { state.write("[]"); }
    syntax::ArraySpecifier::ExplicitlySized(ref e) => {
      state.write("[");
      show_expr(state, &e);
      state.write("]");
    }
  }
}

pub fn show_identifier_and_type(state: &OutputState, ident: &syntax::Identifier, ty: &hir::Type) {
  let _ = write!(state, "{}", ident);

  if !state.output_cxx {
    if let Some(ref arr_spec) = ty.array_sizes {
      show_array_sizes(state, &arr_spec);
    }
  }
}

pub fn show_arrayed_identifier(state: &OutputState, ident: &syntax::ArrayedIdentifier) {
  let _ = write!(state, "{}", ident.ident);

  if let Some(ref arr_spec) = ident.array_spec {
    show_array_spec(state, &arr_spec);
  }
}


pub fn show_array_sizes(state: &OutputState, a: &hir::ArraySizes) {
  state.write("[");
  match &a.sizes[..] {
    [a] => show_hir_expr(state, a),
    _ => panic!()
  }

  state.write("]");
  /*
  match *a {
    syntax::ArraySpecifier::Unsized => { state.write("[]"); }
    syntax::ArraySpecifier::ExplicitlySized(ref e) => {
      state.write("[");
      show_expr(state, &e);
      state.write("]");
    }
  }*/
}

pub fn show_type_qualifier(state: &OutputState, q: &hir::TypeQualifier) {
  let mut qualifiers = q.qualifiers.0.iter();
  let first = qualifiers.next().unwrap();

  show_type_qualifier_spec(state, first);

  for qual_spec in qualifiers {
    state.write(" ");
    show_type_qualifier_spec(state, qual_spec)
  }
}

pub fn show_type_qualifier_spec(state: &OutputState, q: &hir::TypeQualifierSpec) {
  match *q {
    hir::TypeQualifierSpec::Layout(ref l) => show_layout_qualifier(state, &l),
    hir::TypeQualifierSpec::Parameter(ref p) => panic!(),
    hir::TypeQualifierSpec::Memory(ref m) => panic!(),
    hir::TypeQualifierSpec::Invariant => { state.write("invariant"); },
    hir::TypeQualifierSpec::Precise => { state.write("precise"); }
  }
}

pub fn show_syntax_storage_qualifier(state: &OutputState, q: &syntax::StorageQualifier) {
  match *q {
    syntax::StorageQualifier::Const => { state.write("const"); }
    syntax::StorageQualifier::InOut => { state.write("inout"); }
    syntax::StorageQualifier::In => { state.write("in"); }
    syntax::StorageQualifier::Out => { state.write("out"); }
    syntax::StorageQualifier::Centroid => { state.write("centroid"); }
    syntax::StorageQualifier::Patch => { state.write("patch"); }
    syntax::StorageQualifier::Sample => { state.write("sample"); }
    syntax::StorageQualifier::Uniform => { state.write("uniform"); }
    syntax::StorageQualifier::Buffer => { state.write("buffer"); }
    syntax::StorageQualifier::Shared => { state.write("shared"); }
    syntax::StorageQualifier::Coherent => { state.write("coherent"); }
    syntax::StorageQualifier::Volatile => { state.write("volatile"); }
    syntax::StorageQualifier::Restrict => { state.write("restrict"); }
    syntax::StorageQualifier::ReadOnly => { state.write("readonly"); }
    syntax::StorageQualifier::WriteOnly => { state.write("writeonly"); }
    syntax::StorageQualifier::Subroutine(ref n) => show_subroutine(state, &n)
  }
}

pub fn show_subroutine(state: &OutputState, types: &Vec<syntax::TypeName>) {
  state.write("subroutine");

  if !types.is_empty() {
    state.write("(");

    let mut types_iter = types.iter();
    let first = types_iter.next().unwrap();

    show_type_name(state, first);

    for type_name in types_iter {
      state.write(", ");
      show_type_name(state, type_name);
    }

    state.write(")");
  }
}

pub fn show_layout_qualifier(state: &OutputState, l: &syntax::LayoutQualifier) {
  let mut qualifiers = l.ids.0.iter();
  let first = qualifiers.next().unwrap();

  state.write("layout (");
  show_layout_qualifier_spec(state, first);

  for qual_spec in qualifiers {
    state.write(", ");
    show_layout_qualifier_spec(state, qual_spec);
  }

  state.write(")");
}

pub fn show_layout_qualifier_spec(state: &OutputState, l: &syntax::LayoutQualifierSpec) {
  match *l {
    syntax::LayoutQualifierSpec::Identifier(ref i, Some(ref e)) => {
      let _ = write!(state, "{} = ", i);
      show_expr(state, &e);
    }
    syntax::LayoutQualifierSpec::Identifier(ref i, None) => show_identifier(state, &i),
    syntax::LayoutQualifierSpec::Shared => { state.write("shared"); }
  }
}

pub fn show_precision_qualifier(state: &OutputState, p: &syntax::PrecisionQualifier) {
  match *p {
    syntax::PrecisionQualifier::High => { state.write("highp"); }
    syntax::PrecisionQualifier::Medium => { state.write("mediump"); }
    syntax::PrecisionQualifier::Low => { state.write("low"); }
  }
}

pub fn show_interpolation_qualifier(state: &OutputState, i: &syntax::InterpolationQualifier) {
  match *i {
    syntax::InterpolationQualifier::Smooth => { state.write("smooth"); }
    syntax::InterpolationQualifier::Flat => { state.write("flat"); }
    syntax::InterpolationQualifier::NoPerspective => { state.write("noperspective"); }
  }
}

pub fn show_parameter_qualifier(state: &mut OutputState, i: &Option<hir::ParameterQualifier>) {
  if let Some(i) = i {
    if state.output_cxx {
      match *i {
        hir::ParameterQualifier::Out => { state.write("&"); }
        hir::ParameterQualifier::InOut => { state.write("&"); }
        _ => {}
      }
    } else {
      match *i {
        hir::ParameterQualifier::Const => { state.write("const"); }
        hir::ParameterQualifier::In => { state.write("in"); }
        hir::ParameterQualifier::Out => { state.write("out"); }
        hir::ParameterQualifier::InOut => { state.write("inout"); }
      }
    }
  }
}


pub fn show_float(state: &OutputState, x: f32) {
  if x.fract() == 0. {
    write!(state, "{}.", x);
  } else {
    write!(state, "{}", x);
  }
}

pub fn show_double(state: &OutputState, x: f64) {
  if x.fract() == 0. {
    write!(state, "{}.", x);
  } else {
    write!(state, "{}", x);
  }
}

trait SwizzelSelectorExt {
  fn to_args(&self) -> String;
}

impl SwizzelSelectorExt for SwizzleSelector {
  fn to_args(&self) -> String {
    let mut s = Vec::new();
    let fs = match self.field_set {
      hir::FieldSet::Rgba => ["R", "G", "B", "A"],
      hir::FieldSet::Xyzw => ["X", "Y", "Z", "W"],
      hir::FieldSet::Stpq => ["S", "T", "P", "Q"],
    };
    for i in &self.components {
      s.push(fs[*i as usize])
    }
    s.join(", ")
  }
}

fn expr_run_class(state: &OutputState, expr: &hir::Expr) -> hir::RunClass {
  match &expr.kind {
    hir::ExprKind::Variable(i) => symbol_run_class(&state.hir.sym(*i).decl, state.vector_mask),
    hir::ExprKind::IntConst(_) |
    hir::ExprKind::UIntConst(_) |
    hir::ExprKind::BoolConst(_) |
    hir::ExprKind::FloatConst(_) |
    hir::ExprKind::DoubleConst(_) => hir::RunClass::Scalar,
    hir::ExprKind::Unary(_, ref e) => expr_run_class(state, e),
    hir::ExprKind::Binary(_, ref l, ref r) => expr_run_class(state, l).merge(expr_run_class(state, r)),
    hir::ExprKind::Ternary(ref c, ref s, ref e) => expr_run_class(state, c).merge(expr_run_class(state, s)).merge(expr_run_class(state, e)),
    hir::ExprKind::Assignment(ref v, _, ref e) => expr_run_class(state, v).merge(expr_run_class(state, e)),
    hir::ExprKind::Bracket(ref e, ref indx) => expr_run_class(state, e).merge(expr_run_class(state, indx)),
    hir::ExprKind::FunCall(ref fun, ref args) => {
        let arg_mask: u32 = args.iter().enumerate().fold(0, |mask, (idx, e)| {
            if expr_run_class(state, e) == hir::RunClass::Vector {
                mask | (1 << idx)
            } else {
                mask
            }
        });
        match fun {
            hir::FunIdentifier::Identifier(ref sym) => {
                match &state.hir.sym(*sym).decl {
                    hir::SymDecl::NativeFunction(..) => if arg_mask != 0 { hir::RunClass::Vector } else { hir::RunClass::Scalar },
                    hir::SymDecl::UserFunction(ref fd, ref run_class) => {
                        let param_mask: u32 = fd.prototype.parameters.iter().enumerate().fold(arg_mask, |mask, (idx, param)| {
                            if let hir::FunctionParameterDeclaration::Named(Some(qual), p) = param {
                                match qual {
                                    hir::ParameterQualifier::InOut | hir::ParameterQualifier::Out => {
                                        if symbol_run_class(&state.hir.sym(p.sym).decl, arg_mask) == hir::RunClass::Vector {
                                            mask | (1 << idx)
                                        } else {
                                            mask
                                        }
                                    }
                                    _ => mask,
                                }
                            } else {
                                mask
                            }
                        });
                        match *run_class {
                            hir::RunClass::Scalar => hir::RunClass::Scalar,
                            hir::RunClass::Dependent(mask) => {
                                if (mask & param_mask) != 0 {
                                    hir::RunClass::Vector
                                } else {
                                    hir::RunClass::Scalar
                                }
                            }
                            _ => hir::RunClass::Vector,
                        }
                    }
                    hir::SymDecl::Struct(..) => if arg_mask != 0 { hir::RunClass::Vector } else { hir::RunClass::Scalar },
                    _ => panic!(),
                }
            }
            hir::FunIdentifier::Constructor(..) => if arg_mask != 0 { hir::RunClass::Vector } else { hir::RunClass::Scalar },
        }
    }
    hir::ExprKind::Dot(ref e, _) => expr_run_class(state, e),
    hir::ExprKind::SwizzleSelector(ref e, _) => expr_run_class(state, e),
    hir::ExprKind::PostInc(ref e) => expr_run_class(state, e),
    hir::ExprKind::PostDec(ref e) => expr_run_class(state, e),
    hir::ExprKind::Comma(_, ref e) => expr_run_class(state, e),
    hir::ExprKind::Cond(_, ref e) => expr_run_class(state, e),
    hir::ExprKind::CondMask => hir::RunClass::Vector,
  }
}

fn constant_across_all_lanes(e: &hir::Expr, state: &OutputState) -> bool {
  /*
  match &e.kind {
    hir::ExprKind::Variable(i) => {
      match state.hir.sym(*i).decl {
        hir::SymDecl::Variable(storage,..) => {
          match storage {
            hir::StorageClass::Out => return true,
            _ => {}
          }
        }
        _ => { panic!("should be variable") }
      }
    }
    hir::ExprKind::SwizzleSelector(e, ..) => {
      return constant_across_all_lanes(e, state);
    }
    _ => {}
  };*/
  false
}

pub fn show_hir_expr(state: &OutputState, expr: &hir::Expr) {
  show_hir_expr_inner(state, expr, false);
}

pub fn show_hir_expr_inner(state: &OutputState, expr: &hir::Expr, top_level: bool) {
  match expr.kind {
    hir::ExprKind::Variable(ref i) => show_sym(state, i),
    hir::ExprKind::IntConst(ref x) => { let _ = write!(state, "{}", x); }
    hir::ExprKind::UIntConst(ref x) => { let _ = write!(state, "{}u", x); }
    hir::ExprKind::BoolConst(ref x) => { let _ = write!(state, "{}", x); }
    hir::ExprKind::FloatConst(ref x) => show_float(state, *x),
    hir::ExprKind::DoubleConst(ref x) => show_double(state, *x),
    hir::ExprKind::Unary(ref op, ref e) => {
      show_unary_op(state, &op);
      state.write("(");
      show_hir_expr(state, &e);
      state.write(")");
    }
    hir::ExprKind::Binary(ref op, ref l, ref r) => {
      state.write("(");
      show_hir_expr(state, &l);
      state.write(")");
      show_binary_op(state, &op);
      state.write("(");
      show_hir_expr(state, &r);
      state.write(")");
    }
    hir::ExprKind::Ternary(ref c, ref s, ref e) => {
      if state.output_cxx && expr_run_class(state, c) != hir::RunClass::Scalar {
        state.write("if_then_else(");
        show_hir_expr(state, &c);
        state.write(", ");
        show_hir_expr(state, &s);
        state.write(", ");
        show_hir_expr(state, &e);
        state.write(")");
      } else {
        show_hir_expr(state, &c);
        state.write(" ? ");
        show_hir_expr(state, &s);
        state.write(" : ");
        show_hir_expr(state, &e);
      }
    }
    hir::ExprKind::Assignment(ref v, ref op, ref e) => {
      let is_output = hir::is_output(v, &state.hir).is_some();
      let is_scalar_var = expr_run_class(state, v) == hir::RunClass::Scalar;
      let is_scalar_expr = expr_run_class(state, e) == hir::RunClass::Scalar;
      let force_scalar = is_scalar_var && !is_scalar_expr;

      if let Some(mask) = &state.mask {
        let is_scalar_mask = expr_run_class(state, mask) == hir::RunClass::Scalar;
        let force_scalar_mask = is_scalar_var && is_scalar_expr && !is_scalar_mask;

        if force_scalar || force_scalar_mask {
          if top_level {
            state.write("if (");
          } else {
            state.write("(");
          }
        } else {
          state.is_lval.set(true);
          show_hir_expr(state, &v);
          state.is_lval.set(false);
          state.write(" = if_then_else(");
        }

        if is_output && state.return_declared {
          state.write("((");
          show_hir_expr(state, mask);
          state.write(")&ret_mask)");
        } else {
          show_hir_expr(state, mask);
        }
        if force_scalar || force_scalar_mask {
          if top_level {
            state.write("[0]) { ");
          } else {
            state.write("[0] ? ");
          }
          state.is_lval.set(true);
          show_hir_expr(state, &v);
          state.is_lval.set(false);
          state.write(" = ");
        } else {
          state.write(",");
        }

        if op != &syntax::AssignmentOp::Equal {
          show_hir_expr(state, &v);
        }

        match *op {
          syntax::AssignmentOp::Equal => {  }
          syntax::AssignmentOp::Mult => { state.write("*"); }
          syntax::AssignmentOp::Div => { state.write("/"); }
          syntax::AssignmentOp::Mod => { state.write("%"); }
          syntax::AssignmentOp::Add => { state.write("+"); }
          syntax::AssignmentOp::Sub => { state.write("-"); }
          syntax::AssignmentOp::LShift => { state.write("<<"); }
          syntax::AssignmentOp::RShift => { state.write(">>"); }
          syntax::AssignmentOp::And => { state.write("&"); }
          syntax::AssignmentOp::Xor => { state.write("^"); }
          syntax::AssignmentOp::Or => { state.write("|"); }
        }
        if force_scalar {
          state.write("force_scalar(");
        }
        show_hir_expr(state, &e);
        if force_scalar {
          state.write(")");
        }
        if force_scalar || force_scalar_mask {
          if top_level {
            state.write("; }");
          } else {
            state.write(" : ");
            show_hir_expr(state, &v);
            state.write(")");
          }
        } else {
          state.write(",");
          show_hir_expr(state, &v);
          state.write(")");
        }
      } else {
        state.is_lval.set(true);
        show_hir_expr(state, &v);
        state.is_lval.set(false);
        state.write(" ");

        if is_output && state.return_declared {
          state.write("= ");
          if force_scalar {
            state.write("force_scalar(");
          }
          state.write("if_then_else(ret_mask,");

          if op != &syntax::AssignmentOp::Equal {
            show_hir_expr(state, &v);
          }

          match *op {
            syntax::AssignmentOp::Equal => {  }
            syntax::AssignmentOp::Mult => { state.write("*"); }
            syntax::AssignmentOp::Div => { state.write("/"); }
            syntax::AssignmentOp::Mod => { state.write("%"); }
            syntax::AssignmentOp::Add => { state.write("+"); }
            syntax::AssignmentOp::Sub => { state.write("-"); }
            syntax::AssignmentOp::LShift => { state.write("<<"); }
            syntax::AssignmentOp::RShift => { state.write(">>"); }
            syntax::AssignmentOp::And => { state.write("&"); }
            syntax::AssignmentOp::Xor => { state.write("^"); }
            syntax::AssignmentOp::Or => { state.write("|"); }
          }
          show_hir_expr(state, &e);
          state.write(",");
          show_hir_expr(state, &v);
          state.write(")");
        } else {
          show_assignment_op(state, &op);
          state.write(" ");
          if force_scalar {
            state.write("force_scalar(");
          }
          show_hir_expr(state, &e);
        }

        if force_scalar {
          state.write(")");
        }
      }
    }
    hir::ExprKind::Bracket(ref e, ref indx) => {
      show_hir_expr(state, &e);
      state.write("[");
      show_hir_expr(state, indx);
      state.write("]");
    }
    hir::ExprKind::FunCall(ref fun, ref args) => {
      let mut cond_mask: u32 = 0;
      let mut adapt_mask: u32 = 0;
      let mut has_ret = false;
      let mut array_constructor = false;

      let mut arg_mask: u32 = 0;
      for (idx, e) in args.iter().enumerate() {
        if expr_run_class(state, e) == hir::RunClass::Vector {
          arg_mask |= 1 << idx;
        }
      }

      match fun {
        hir::FunIdentifier::Constructor(t) => {
          let is_scalar = state.is_scalar.replace(arg_mask == 0);
          show_type(state, t);
          state.is_scalar.set(is_scalar);
          array_constructor = t.array_sizes.is_some();
        }
        hir::FunIdentifier::Identifier(name) => {
          if state.output_cxx {
            match &state.hir.sym(*name).decl {
            hir::SymDecl::NativeFunction(..) => show_sym(state, name),
            hir::SymDecl::UserFunction(ref fd, ref run_class) => {
              if (state.mask.is_some() || state.return_declared) && !fd.globals.is_empty() {
                cond_mask |= 1 << 31;
              }
              let mut param_mask: u32 = 0;
              for (idx, (param, e)) in fd.prototype.parameters.iter().zip(args.iter()).enumerate() {
                if let hir::FunctionParameterDeclaration::Named(qual, p) = param {
                  if symbol_run_class(&state.hir.sym(p.sym).decl, arg_mask) == hir::RunClass::Vector {
                    param_mask |= 1 << idx;
                  }
                  match qual {
                    Some(hir::ParameterQualifier::InOut) | Some(hir::ParameterQualifier::Out) => {
                      if state.mask.is_some() || state.return_declared {
                        cond_mask |= 1 << idx;
                      }
                      if (!arg_mask & param_mask & (1 << idx)) != 0 {
                        if adapt_mask == 0 {
                          state.write(if top_level { "{ " } else { "({ " });
                        }
                        show_type(state, &p.ty);
                        write!(state, " _arg{}_ = ", idx);
                        show_hir_expr(state, e);
                        state.write("; ");
                        adapt_mask |= 1 << idx;
                      }
                    }
                    _ => {}
                  }
                }
              }
              if adapt_mask != 0 && fd.prototype.ty.kind != hir::TypeKind::Void && !top_level {
                state.write("auto _ret_ = ");
                has_ret = true;
              }
              show_sym(state, name);
              let mut deps = state.deps.borrow_mut();
              let dep_key = (*name, if cond_mask != 0 { param_mask | (1 << 31) } else { param_mask });
              if !deps.contains(&dep_key) {
                deps.push(dep_key);
              }
            }
            hir::SymDecl::Struct(..) => {
              show_sym(state, name);
              if arg_mask == 0 {
                state.write("_scalar");
              }
            }
            _ => panic!("bad identifier to function call"),
            }
          }
        }
        _ => {}
      }

      if array_constructor {
        state.write("{");
      } else {
        state.write("(");
      }

      for (idx, e) in args.iter().enumerate() {
        if idx != 0 {
          state.write(", ");
        }
        if (adapt_mask & (1 << idx)) != 0 {
          write!(state, "_arg{}_", idx);
        } else {
          show_hir_expr(state, e);
        }
      }

      if cond_mask != 0 {
        if !args.is_empty() {
          state.write(", ");
        }
        if let Some(mask) = &state.mask {
          if state.return_declared {
            state.write("(");
            show_hir_expr(state, mask);
            state.write(")&ret_mask");
          } else {
            show_hir_expr(state, mask);
          }
        } else if state.return_declared {
          state.write("ret_mask");
        } else {
          state.write("~0");
        }
      }

      if array_constructor {
        state.write("}");
      } else {
        state.write(")");
      }

      if adapt_mask != 0 {
        state.write("; ");
        for (idx, e) in args.iter().enumerate() {
          if (adapt_mask & (1 << idx)) != 0 {
            state.is_lval.set(true);
            show_hir_expr(state, e);
            state.is_lval.set(false);
            write!(state, " = force_scalar(_arg{}_); ", idx);
          }
        }
        if has_ret {
          state.write("_ret_; })");
        } else {
          state.write(if top_level { "}" } else { "})" });
        }
      }
    }
    hir::ExprKind::Dot(ref e, ref i) => {
      state.write("(");
      show_hir_expr(state, &e);
      state.write(")");
      state.write(".");
      show_identifier(state, i);
    }
    hir::ExprKind::SwizzleSelector(ref e, ref s) => {
      if state.output_cxx {
        state.write("(");
        show_hir_expr(state, &e);
        if state.is_lval.get() && s.components.len() > 1 {
          state.write(").lsel(");
        } else {
          state.write(").sel(");
        }
        state.write(&s.to_args());
        state.write(")");
      } else {
        state.write("(");
        show_hir_expr(state, &e);
        state.write(")");
        state.write(".");
        state.write(&s.to_string());
      }
    }
    hir::ExprKind::PostInc(ref e) => {
      show_hir_expr(state, &e);
      state.write("++");
    }
    hir::ExprKind::PostDec(ref e) => {
      show_hir_expr(state, &e);
      state.write("--");
    }
    hir::ExprKind::Comma(ref a, ref b) => {
      show_hir_expr(state, &a);
      state.write(", ");
      show_hir_expr(state, &b);
    }
    hir::ExprKind::Cond(index, _) => {
      write!(state, "_c{}_", index);
    }
    hir::ExprKind::CondMask => {
      state.write("_cond_mask_");
    }
  }
}

pub fn show_expr(state: &OutputState, expr: &syntax::Expr) {
  match *expr {
    syntax::Expr::Variable(ref i) => show_identifier(state, &i),
    syntax::Expr::IntConst(ref x) => { let _ = write!(state, "{}", x); }
    syntax::Expr::UIntConst(ref x) => { let _ = write!(state, "{}u", x); }
    syntax::Expr::BoolConst(ref x) => { let _ = write!(state, "{}", x); }
    syntax::Expr::FloatConst(ref x) => show_float(state, *x),
    syntax::Expr::DoubleConst(ref x) => show_double(state, *x),
    syntax::Expr::Unary(ref op, ref e) => {
      show_unary_op(state, &op);
      state.write("(");
      show_expr(state, &e);
      state.write(")");
    }
    syntax::Expr::Binary(ref op, ref l, ref r) => {
      state.write("(");
      show_expr(state, &l);
      state.write(")");
      show_binary_op(state, &op);
      state.write("(");
      show_expr(state, &r);
      state.write(")");
    }
    syntax::Expr::Ternary(ref c, ref s, ref e) => {
      show_expr(state, &c);
      state.write(" ? ");
      show_expr(state, &s);
      state.write(" : ");
      show_expr(state, &e);
    }
    syntax::Expr::Assignment(ref v, ref op, ref e) => {
      show_expr(state, &v);
      state.write(" ");
      show_assignment_op(state, &op);
      state.write(" ");
      show_expr(state, &e);
    }
    syntax::Expr::Bracket(ref e, ref a) => {
      show_expr(state, &e);
      show_array_spec(state, &a);
    }
    syntax::Expr::FunCall(ref fun, ref args) => {
      show_function_identifier(state, &fun);
      state.write("(");

      if !args.is_empty() {
        let mut args_iter = args.iter();
        let first = args_iter.next().unwrap();
        show_expr(state, first);

        for e in args_iter {
          state.write(", ");
          show_expr(state, e);
        }
      }

      state.write(")");
    }
    syntax::Expr::Dot(ref e, ref i) => {
      state.write("(");
      show_expr(state, &e);
      state.write(")");
      state.write(".");
      show_identifier(state, &i);
    }
    syntax::Expr::PostInc(ref e) => {
      show_expr(state, &e);
      state.write("++");
    }
    syntax::Expr::PostDec(ref e) => {
      show_expr(state, &e);
      state.write("--");
    }
    syntax::Expr::Comma(ref a, ref b) => {
      show_expr(state, &a);
      state.write(", ");
      show_expr(state, &b);
    }
  }
}

pub fn show_unary_op(state: &OutputState, op: &syntax::UnaryOp) {
  match *op {
    syntax::UnaryOp::Inc => { state.write("++"); }
    syntax::UnaryOp::Dec => { state.write("--"); }
    syntax::UnaryOp::Add => { state.write("+"); }
    syntax::UnaryOp::Minus => { state.write("-"); }
    syntax::UnaryOp::Not => { state.write("!"); }
    syntax::UnaryOp::Complement => { state.write("~"); }
  }
}

pub fn show_binary_op(state: &OutputState, op: &syntax::BinaryOp) {
  match *op {
    syntax::BinaryOp::Or => { state.write("||"); }
    syntax::BinaryOp::Xor => { state.write("^^"); }
    syntax::BinaryOp::And => { state.write("&&"); }
    syntax::BinaryOp::BitOr => { state.write("|"); }
    syntax::BinaryOp::BitXor => { state.write("^"); }
    syntax::BinaryOp::BitAnd => { state.write("&"); }
    syntax::BinaryOp::Equal => { state.write("=="); }
    syntax::BinaryOp::NonEqual => { state.write("!="); }
    syntax::BinaryOp::LT => { state.write("<"); }
    syntax::BinaryOp::GT => { state.write(">"); }
    syntax::BinaryOp::LTE => { state.write("<="); }
    syntax::BinaryOp::GTE => { state.write(">="); }
    syntax::BinaryOp::LShift => { state.write("<<"); }
    syntax::BinaryOp::RShift => { state.write(">>"); }
    syntax::BinaryOp::Add => { state.write("+"); }
    syntax::BinaryOp::Sub => { state.write("-"); }
    syntax::BinaryOp::Mult => { state.write("*"); }
    syntax::BinaryOp::Div => { state.write("/"); }
    syntax::BinaryOp::Mod => { state.write("%"); }
  }
}

pub fn show_assignment_op(state: &OutputState, op: &syntax::AssignmentOp) {
  match *op {
    syntax::AssignmentOp::Equal => { state.write("="); }
    syntax::AssignmentOp::Mult => { state.write("*="); }
    syntax::AssignmentOp::Div => { state.write("/="); }
    syntax::AssignmentOp::Mod => { state.write("%="); }
    syntax::AssignmentOp::Add => { state.write("+="); }
    syntax::AssignmentOp::Sub => { state.write("-="); }
    syntax::AssignmentOp::LShift => { state.write("<<="); }
    syntax::AssignmentOp::RShift => { state.write(">>="); }
    syntax::AssignmentOp::And => { state.write("&="); }
    syntax::AssignmentOp::Xor => { state.write("^="); }
    syntax::AssignmentOp::Or => { state.write("|="); }
  }
}

pub fn show_function_identifier(state: &OutputState, i: &syntax::FunIdentifier) {
  match *i {
    syntax::FunIdentifier::Identifier(ref n) => show_identifier(state, &n),
    syntax::FunIdentifier::Expr(ref e) => show_expr(state, &*e)
  }
}

pub fn show_hir_function_identifier(state: &OutputState, i: &hir::FunIdentifier) {
  match *i {
    hir::FunIdentifier::Identifier(ref n) => show_sym(state, n),
    hir::FunIdentifier::Constructor(ref t) => show_type(state, &*t),
  }
}

pub fn show_declaration(state: &mut OutputState, d: &hir::Declaration) {
  show_indent(state);
  match *d {
    hir::Declaration::FunctionPrototype(ref proto) => {
      if !state.output_cxx {
        show_function_prototype(state, &proto);
        state.write(";\n");
      }
    }
    hir::Declaration::InitDeclaratorList(ref list) => {
      show_init_declarator_list(state, &list);
      state.write(";\n");
    }
    hir::Declaration::Precision(ref qual, ref ty) => {
      if !state.output_cxx {
        show_precision_qualifier(state, &qual);
        show_type_specifier(state, &ty);
        state.write(";\n");
      }
    }
    hir::Declaration::Block(ref block) => {
      show_block(state, &block);
      state.write(";\n");
    }
    hir::Declaration::Global(ref qual, ref identifiers) => {
      show_type_qualifier(state, &qual);

      if !identifiers.is_empty() {
        let mut iter = identifiers.iter();
        let first = iter.next().unwrap();
        show_identifier(state, first);

        for identifier in iter {
          let _ = write!(state, ", {}", identifier);
        }
      }

      state.write(";\n");
    }
    hir::Declaration::StructDefinition(ref sym) => {
      show_sym_decl(state, sym);

      state.write(";\n");
    }
  }
}

pub fn show_function_prototype(state: &mut OutputState, fp: &hir::FunctionPrototype) {
  let is_scalar = state.is_scalar.replace(!state.return_vector);
  show_type(state, &fp.ty);
  state.is_scalar.set(is_scalar);

  state.write(" ");
  show_identifier(state, &fp.name);

  state.write("(");

  if !fp.parameters.is_empty() {
    let mut iter = fp.parameters.iter();
    let first = iter.next().unwrap();
    show_function_parameter_declaration(state, first);

    for param in iter {
      state.write(", ");
      show_function_parameter_declaration(state, param);
    }
  }

  if state.output_cxx && (state.vector_mask & (1 << 31)) != 0 {
    if !fp.parameters.is_empty() {
      state.write(", ");
    }
    state.write("I32 _cond_mask_");
  }

  state.write(")");
}

pub fn show_function_parameter_declaration(state: &mut OutputState, p: &hir::FunctionParameterDeclaration) {
  match *p {
    hir::FunctionParameterDeclaration::Named(ref qual, ref fpd) => {
      if state.output_cxx {
        let is_scalar = state.is_scalar.replace(symbol_run_class(&state.hir.sym(fpd.sym).decl, state.vector_mask) == hir::RunClass::Scalar);
        show_type(state, &fpd.ty);
        state.is_scalar.set(is_scalar);
        show_parameter_qualifier(state, qual);
      } else {
        show_parameter_qualifier(state, qual);
        state.write(" ");
        show_type(state, &fpd.ty);
      }
      state.write(" ");
      show_identifier_and_type(state, &fpd.name, &fpd.ty);
    }
    hir::FunctionParameterDeclaration::Unnamed(ref qual, ref ty) => {
      if state.output_cxx {
        show_type_specifier(state, ty);
        show_parameter_qualifier(state, qual);
      } else {
        show_parameter_qualifier(state, qual);
        state.write(" ");
        show_type_specifier(state, ty);
      }
    }
  }
}

pub fn show_init_declarator_list(state: &mut OutputState, i: &hir::InitDeclaratorList) {
  show_single_declaration(state, &i.head);

  for decl in &i.tail {
    state.write(", ");
    show_single_declaration_no_type(state, decl);
  }
}

pub fn show_single_declaration(state: &mut OutputState, d: &hir::SingleDeclaration) {
  if state.output_cxx {
    show_single_declaration_cxx(state, d)
  } else {
    show_single_declaration_glsl(state, d)
  }
}


pub fn show_single_declaration_glsl(state: &mut OutputState, d: &hir::SingleDeclaration) {
  if let Some(ref qual) = d.qualifier {
    show_type_qualifier(state, &qual);
    state.write(" ");
  }

  let sym = state.hir.sym(d.name);
  match &sym.decl {
    hir::SymDecl::Global(storage, interpolation, ..) => {
      show_storage_class(state, storage);
      if let Some(i) = interpolation {
        show_interpolation_qualifier(state, i);
      }
    }
    hir::SymDecl::Local(storage, ..) => {
      show_storage_class(state, storage)
    }
    _ => panic!("should be variable")
  }

  if let Some(ty_def) = d.ty_def {
    show_sym_decl(state, &ty_def);
  } else {
    show_type(state, &d.ty);
  }

  state.write(" ");
  state.write(sym.name.as_str());

  if let Some(ref arr_spec) = d.ty.array_sizes {
    show_array_sizes(state, &arr_spec);
  }

  if let Some(ref initializer) = d.initializer {
    state.write(" = ");
    show_initializer(state, initializer);
  }
}



fn symbol_run_class(decl: &hir::SymDecl, vector_mask: u32) -> hir::RunClass {
  let run_class = match decl {
    hir::SymDecl::Global(_, _, _, run_class) => *run_class,
    hir::SymDecl::Local(_, _, run_class) => *run_class,
    _ => hir::RunClass::Vector,
  };
  match run_class {
    hir::RunClass::Scalar => hir::RunClass::Scalar,
    hir::RunClass::Dependent(mask) => {
        if (mask & vector_mask) != 0 {
            hir::RunClass::Vector
        } else {
            hir::RunClass::Scalar
        }
    }
    _ => hir::RunClass::Vector,
  }
}

pub fn show_single_declaration_cxx(state: &mut OutputState, d: &hir::SingleDeclaration) {
  let sym = state.hir.sym(d.name);
  if state.kind == ShaderKind::Fragment {
    match &sym.decl {
      hir::SymDecl::Global(hir::StorageClass::Out, ..) => {
          write!(state, "#define {} gl_FragColor\n", sym.name);
          show_indent(state);
          state.write("// ");
      }
      _ => {}
    }
  }
  let is_scalar = state.is_scalar.replace(symbol_run_class(&sym.decl, state.vector_mask) == hir::RunClass::Scalar);

  if let Some(ref array) = d.ty.array_sizes {
    show_type(state, &d.ty);
  } else {
    if let Some(ty_def) = d.ty_def {
      show_sym_decl(state, &ty_def);
    } else {
      show_type(state, &d.ty);
    }
  }

  // XXX: this is pretty grotty
  state.write(" ");
  show_sym_decl(state, &d.name);

  state.is_scalar.set(false);

  if let Some(ref initializer) = d.initializer {
    state.write(" = ");
    show_initializer(state, initializer);
  }
}

pub fn show_single_declaration_no_type(state: &OutputState, d: &hir::SingleDeclarationNoType) {
  show_arrayed_identifier(state, &d.ident);

  if let Some(ref initializer) = d.initializer {
    state.write(" = ");
    show_initializer(state, initializer);
  }
}

pub fn show_initializer(state: &OutputState, i: &hir::Initializer) {
  match *i {
    hir::Initializer::Simple(ref e) => show_hir_expr(state, e),
    hir::Initializer::List(ref list) => {
      let mut iter = list.0.iter();
      let first = iter.next().unwrap();

      state.write("{ ");
      show_initializer(state, first);

      for ini in iter {
        state.write(", ");
        show_initializer(state, ini);
      }

      state.write(" }");
    }
  }
}

pub fn show_block(state: &mut OutputState, b: &hir::Block) {
  show_type_qualifier(state, &b.qualifier);
  state.write(" ");
  show_identifier(state, &b.name);
  state.write(" {");

  for field in &b.fields {
    panic!();//show_struct_field(state, field);
    state.write("\n");
  }
  state.write("}");

  if let Some(ref ident) = b.identifier {
    panic!();//show_arrayed_identifier(state, ident);
  }
}

// This is a hack to run through the first time with an empty writter to find if 'return' is declared.
pub fn has_conditional_return(state: &mut OutputState, cst: &hir::CompoundStatement) -> bool {
  let buffer = state.push_buffer();
  show_compound_statement(state, cst);
  state.pop_buffer(buffer);
  let result = state.return_declared;
  state.return_declared = false;
  result
}

pub fn show_function_definition(state: &mut OutputState, fd: &hir::FunctionDefinition, vector_mask: u32, run_class: hir::RunClass) {
//  println!("start {:?} {:?}", fd.prototype.name, vector_mask);
  if state.output_cxx && fd.prototype.name.as_str() == "main" {
    state.write("ALWAYS_INLINE ");
  }
  show_function_prototype(state, &fd.prototype);
  state.write(" ");
  state.return_type = Some(Box::new(fd.prototype.ty.clone()));

  if state.output_cxx && (vector_mask & (1 << 31)) != 0 {
    state.mask = Some(Box::new(hir::Expr { kind: hir::ExprKind::CondMask, ty: hir::Type::new(hir::TypeKind::Bool) }));
  }

  show_indent(state);
  state.write("{\n");

  state.indent();
  if has_conditional_return(state, &fd.body) {
    show_indent(state);
    state.write(if state.return_vector { "I32" } else { "int32_t" });
    state.write(" ret_mask = ");
    if let Some(mask) = &state.mask {
        show_hir_expr(state, mask);
    } else {
        state.write("~0");
    }
    state.write(";\n");
    // XXX: the cloning here is bad
    show_indent(state);
    if fd.prototype.ty != Type::new(hir::TypeKind::Void) {
      let is_scalar = state.is_scalar.replace(!state.return_vector);
      show_type(state, &state.return_type.clone().unwrap());
      state.write(" ret;\n");
      state.is_scalar.set(is_scalar);
    }
  }
  for st in &fd.body.statement_list {
    show_statement(state, st);
  }
  if state.return_declared {
    show_indent(state);
    if fd.prototype.ty == Type::new(hir::TypeKind::Void) {
      state.write("return;\n");
    } else {
      state.write("return ret;\n");
    }
  }
  state.outdent();

  show_indent(state);
  state.write("}\n");
  // println!("end {:?}", fd.prototype.name);

  state.return_type = None;
  state.return_declared = false;
  state.mask = None;
}

pub fn show_compound_statement(state: &mut OutputState, cst: &hir::CompoundStatement) {
  show_indent(state);
  state.write("{\n");

  state.indent();
  for st in &cst.statement_list {
    show_statement(state, st);
  }
  state.outdent();

  show_indent(state);
  state.write("}\n");
}

pub fn show_statement(state: &mut OutputState, st: &hir::Statement) {
  match *st {
    hir::Statement::Compound(ref cst) => show_compound_statement(state, cst),
    hir::Statement::Simple(ref sst) => show_simple_statement(state, sst)
  }
}

pub fn show_simple_statement(state: &mut OutputState, sst: &hir::SimpleStatement) {
  match *sst {
    hir::SimpleStatement::Declaration(ref d) => show_declaration(state, d),
    hir::SimpleStatement::Expression(ref e) => show_expression_statement(state, e),
    hir::SimpleStatement::Selection(ref s) => show_selection_statement(state, s),
    hir::SimpleStatement::Switch(ref s) => show_switch_statement(state, s),
    hir::SimpleStatement::Iteration(ref i) => show_iteration_statement(state, i),
    hir::SimpleStatement::Jump(ref j) => show_jump_statement(state, j)
  }
}

pub fn show_indent(state: &OutputState) {
  for i in 0..state.indent {
    state.write(" ");
  }
}

pub fn show_expression_statement(state: &mut OutputState, est: &hir::ExprStatement) {
  show_indent(state);

  if let Some(ref e) = *est {
    show_hir_expr_inner(state, e, true);
  }

  state.write(";\n");
}

pub fn show_selection_statement(state: &mut OutputState, sst: &hir::SelectionStatement) {
  show_indent(state);

  if state.output_cxx && (state.return_declared || expr_run_class(state, &sst.cond) != hir::RunClass::Scalar) {
    let (cond_index, mask) = if state.mask.is_none() || sst.else_stmt.is_some() {
        let cond = sst.cond.clone();
        state.cond_index += 1;
        let cond_index = state.cond_index;
        write!(state, "auto _c{}_ = ", cond_index);
        show_hir_expr(state, &cond);
        state.write(";\n");
        (cond_index, Box::new(hir::Expr { kind: hir::ExprKind::Cond(cond_index, cond), ty: hir::Type::new(hir::TypeKind::Bool) }))
    } else {
        (0, sst.cond.clone())
    };

    let previous = mem::replace(&mut state.mask, None);
    state.mask = Some(match previous.clone() {
      Some(e) => {
        let cond = Box::new(hir::Expr {
          kind: hir::ExprKind::Binary(syntax::BinaryOp::BitAnd, e, mask.clone()),
          ty: hir::Type::new(hir::TypeKind::Bool)
        });
        state.cond_index += 1;
        let nested_cond_index = state.cond_index;
        show_indent(state);
        write!(state, "auto _c{}_ = ", nested_cond_index);
        show_hir_expr(state, &cond);
        state.write(";\n");
        Box::new(hir::Expr { kind: hir::ExprKind::Cond(nested_cond_index, cond), ty: hir::Type::new(hir::TypeKind::Bool) })
      }
      None => mask.clone()
    });

    show_statement(state, &sst.body);
    state.mask = previous;

    if let Some(rest) = &sst.else_stmt {
      let previous = state.mask.clone();
      // invert the condition
      let inverted_cond =
          Box::new(hir::Expr {
            kind: hir::ExprKind::Unary(UnaryOp::Complement, mask),
            ty: hir::Type::new(hir::TypeKind::Bool),
          });
      let previous = mem::replace(&mut state.mask, None);
      state.mask = Some(match previous.clone() {
        Some(e) => {
          let cond = Box::new(hir::Expr {
            kind: hir::ExprKind::Binary(syntax::BinaryOp::BitAnd, e, inverted_cond),
            ty: hir::Type::new(hir::TypeKind::Bool)
          });
          show_indent(state);
          write!(state, "_c{}_ = ", cond_index);
          show_hir_expr(state, &cond);
          state.write(";\n");
          Box::new(hir::Expr { kind: hir::ExprKind::Cond(cond_index, cond), ty: hir::Type::new(hir::TypeKind::Bool) })
        }
        None => inverted_cond
      });

      show_statement(state, rest);
      state.mask = previous;
    }
  } else {
    state.write("if (");
    show_hir_expr(state, &sst.cond);
    state.write(") {\n");

    state.indent();
    show_statement(state, &sst.body);
    state.outdent();

    show_indent(state);
    if let Some(rest) = &sst.else_stmt {
      state.write("} else ");
      show_statement(state, rest);
    } else {
      state.write("}\n");
    }
  }
}

fn case_stmts_to_if_stmts(stmts: &Vec<Statement>, last: bool) -> (Option<Box<Statement>>, bool) {
  // Look for jump statements and remove them
  // We currently are pretty strict on the form that the statement
  // list needs to be in. This can be loosened as needed.
  let mut fallthrough = false;
  let cstmt = match &stmts[..] {
    [hir::Statement::Compound(c)] => {
      match c.statement_list.split_last() {
        Some((hir::Statement::Simple(s), rest)) => {
          match **s {
            hir::SimpleStatement::Jump(hir::JumpStatement::Break) => {
              hir::CompoundStatement{statement_list: rest.to_owned() }
            }
            _ => panic!("fall through not supported")
          }
        }
        _ => panic!("empty compound")
      }
    },
    [hir::Statement::Simple(s)] => {
      match **s {
        hir::SimpleStatement::Jump(hir::JumpStatement::Break) => {
          hir::CompoundStatement{statement_list: Vec::new() }
        }
        _ => {
          if last {
            // we don't need a break at the end
            hir::CompoundStatement{statement_list: vec![hir::Statement::Simple(s.clone())] }
          } else {
            panic!("fall through not supported {:?}", s)
          }
        }
      }
    }
    [] => {
      return (None, true)
    }
    stmts => {
      match stmts.split_last() {
        Some((hir::Statement::Simple(s), rest)) => {
          match **s {
            hir::SimpleStatement::Jump(hir::JumpStatement::Break) => {
              hir::CompoundStatement { statement_list: rest.to_owned() }
            }
            _ => {
              if !last {
                fallthrough = true;
              }
              hir::CompoundStatement { statement_list: stmts.to_owned() }
            }
          }
        }
        _ => panic!("unexpected empty")
      }
    }
  };
  let stmts = Box::new(hir::Statement::Compound(Box::new(cstmt)));
  (Some(stmts), fallthrough)
}


fn build_selection<'a, I: Iterator<Item = &'a hir::Case>>(
  head: &Box<hir::Expr>,
  case: &hir::Case,
  mut cases: I,
  default: Option<&hir::Case>,
  previous_condition: Option<Box<hir::Expr>>,
  previous_stmts: Option<Box<hir::Statement>>
) -> hir::SelectionStatement {

  let cond = match &case.label {
    hir::CaseLabel::Case(e) => {
      Some(Box::new(hir::Expr { kind: hir::ExprKind::Binary(syntax::BinaryOp::Equal, head.clone(), e.clone()),
        ty: hir::Type::new(hir::TypeKind::Bool)}))
    }
    hir::CaseLabel::Def => None
  };

  // if we have two conditions join them
  let cond = match (&previous_condition, &cond) {
    (Some(prev), Some(cond)) => Some(Box::new(hir::Expr {
      kind: hir::ExprKind::Binary(syntax::BinaryOp::Or, prev.clone(), cond.clone()),
      ty: hir::Type::new(hir::TypeKind::Bool)
    })),
    (_, cond) => cond.clone()
  };

  /*

  // find the next case that's not a default
  let next_case = loop {
    match cases.next() {
      Some(hir::Case { label: hir::CaseLabel::Def, ..}) => { },
      case => break case,
    }
  };*/

  let (cond, body, else_stmt) = match (cond, cases.next()) {
    (None, Some(next_case)) => {
      assert!(previous_stmts.is_none());
      // default so just move on to the next
      return build_selection(head, next_case, cases, default, None, None)
    },
    (Some(cond), Some(next_case)) => {
      assert!(previous_stmts.is_none());
      let (stmts, fallthrough) = case_stmts_to_if_stmts(&case.stmts, false);
      if !fallthrough && stmts.is_some() {
          (cond, stmts.unwrap(), Some(Box::new(
            hir::Statement::Simple(
              Box::new(
                hir::SimpleStatement::Selection(
                  build_selection(head, next_case, cases, default, None, None)))))))
      } else {
          // empty so fall through to the next
          return build_selection(head, next_case, cases, default, Some(cond), stmts);
      }
    }
    (Some(cond), None) => {
      // non-default last
      assert!(previous_stmts.is_none());
      let (stmts, _) = case_stmts_to_if_stmts(&case.stmts, default.is_none());
      let stmts = stmts.expect("empty case labels unsupported at the end");
      // add the default case at the end if we have one
      (cond, stmts, match default {
        Some(default) => {
          let (default_stmts, fallthrough) = case_stmts_to_if_stmts(&default.stmts, true);
          assert!(!fallthrough);
          Some(default_stmts.expect("empty default unsupported"))
        },
        None => None
      })
    }
    (None, None) => {
      // default, last

      assert!(default.is_some());

      let (stmts, fallthrough) = case_stmts_to_if_stmts(&case.stmts, true);
      let stmts = stmts.expect("empty default unsupported");
      assert!(!fallthrough);

      match previous_stmts {
        Some(previous_stmts) => {
          let cond = previous_condition.expect("must have previous condition");
          (cond, previous_stmts, Some(stmts))
        }
        None => {
          let cond = Box::new(hir::Expr {
            kind: hir::ExprKind::BoolConst(true),
            ty: hir::Type::new(hir::TypeKind::Bool)
          });
          (cond, stmts, None)
        }
      }
    }
  };

  hir::SelectionStatement {
    cond,
    body,
    else_stmt
  }
}

pub fn lower_switch_to_ifs(sst: &hir::SwitchStatement) -> hir::SelectionStatement {
  let default = sst.cases.iter().find(|x| x.label == hir::CaseLabel::Def);
  let mut cases = sst.cases.iter();
  let r = build_selection(&sst.head, cases.next().unwrap(), cases, default, None, None);
  r
}

fn is_declaration(stmt: &hir::Statement) -> bool {
  if let hir::Statement::Simple(s) = stmt {
    if let hir::SimpleStatement::Declaration(..) = **s {
      return true;
    }
  }
  return false;
}

pub fn show_switch_statement(state: &mut OutputState, sst: &hir::SwitchStatement) {
  if state.output_cxx && expr_run_class(state, &sst.head) != hir::RunClass::Scalar {
    // XXX: when lowering switches we end up with a mask that has
    // a bunch of mutually exclusive conditions.
    // It would be nice if we could fold them together.
    let ifs = lower_switch_to_ifs(sst);
    return show_selection_statement(state, &ifs);
  }

  show_indent(state);
  state.write("switch (");
  show_hir_expr(state, &sst.head);
  state.write(") {\n");
  state.indent();

  for case in &sst.cases {
    show_case_label(state, &case.label);
    state.indent();

    let has_declaration = case.stmts.iter().any(|x| is_declaration(x));
    // glsl allows declarations in switch statements while C requires them to be
    // in a compound statement. If we have a declaration wrap the statements in an block.
    // This will break some glsl shaders but keeps the saner ones working
    if has_declaration {
      show_indent(state);
      state.write("{\n");
      state.indent();
    }
    for st in &case.stmts {
      show_statement(state, st);
    }

    if has_declaration {
      show_indent(state);
      state.write("}\n");
      state.outdent();
    }

    state.outdent();
  }
  state.outdent();
  show_indent(state);
  state.write("}\n");

}

pub fn show_case_label(state: &mut OutputState, cl: &hir::CaseLabel) {
  show_indent(state);
  match *cl {
    hir::CaseLabel::Case(ref e) => {
      state.write("case ");
      show_hir_expr(state, e);
      state.write(":\n");
    }
    hir::CaseLabel::Def => { state.write("default:\n"); }
  }
}

pub fn show_iteration_statement(state: &mut OutputState, ist: &hir::IterationStatement) {
  show_indent(state);
  match *ist {
    hir::IterationStatement::While(ref cond, ref body) => {
      state.write("while (");
      show_condition(state, cond);
      state.write(") ");
      show_statement(state, body);
    }
    hir::IterationStatement::DoWhile(ref body, ref cond) => {
      state.write("do ");
      show_statement(state, body);
      state.write(" while (");
      show_hir_expr(state, cond);
      state.write(")\n");
    }
    hir::IterationStatement::For(ref init, ref rest, ref body) => {
      state.write("for (");
      show_for_init_statement(state, init);
      show_for_rest_statement(state, rest);
      state.write(") ");
      show_statement(state, body);
    }
  }
}

pub fn show_condition(state: &mut OutputState, c: &hir::Condition) {
  match *c {
    hir::Condition::Expr(ref e) => show_hir_expr(state, e),
    /*hir::Condition::Assignment(ref ty, ref name, ref initializer) => {
      show_type(state, ty);
      state.write(" ");
      show_identifier(f, name);
      state.write(" = ");
      show_initializer(state, initializer);
    }*/
  }
}

pub fn show_for_init_statement(state: &mut OutputState, i: &hir::ForInitStatement) {
  match *i {
    hir::ForInitStatement::Expression(ref expr) => {
      if let Some(ref e) = *expr {
        show_hir_expr(state, e);
      }
    }
    hir::ForInitStatement::Declaration(ref d) => {
      show_declaration(state, d);
    }
  }
}

pub fn show_for_rest_statement(state: &mut OutputState, r: &hir::ForRestStatement) {
  if let Some(ref cond) = r.condition {
    show_condition(state, cond);
  }

  state.write("; ");

  if let Some(ref e) = r.post_expr {
    show_hir_expr(state, e);
  }
}

fn use_return_mask(state: &OutputState) -> bool {
  if let Some(mask) = &state.mask {
    mask.kind != hir::ExprKind::CondMask
  } else {
    false
  }
}

pub fn show_jump_statement(state: &mut OutputState, j: &hir::JumpStatement) {
  show_indent(state);
  match *j {
    hir::JumpStatement::Continue => { state.write("continue;\n"); }
    hir::JumpStatement::Break => { state.write("break;\n"); }
    hir::JumpStatement::Discard => {
      if state.output_cxx {
        state.uses_discard = true;
        if let Some(mask) = &state.mask {
          state.write("isPixelDiscarded |= (");
          show_hir_expr(state, mask);
          state.write(")");
          if state.return_declared {
            state.write("&ret_mask");
          }
          state.write(";\n");
        } else {
          state.write("isPixelDiscarded = true;\n");
        }
      } else {
        state.write("discard;\n");
      }
    }
    hir::JumpStatement::Return(ref e) => {
      if let Some(e) = e {
        if state.output_cxx {
          if use_return_mask(state) {
            // We cast any conditions by `ret_mask_type` so that scalars nicely
            // convert to -1. i.e. I32 &= bool will give the wrong result. while I32 &= I32(bool) works
            let ret_mask_type = if state.return_vector { "I32" } else { "int32_t" };
            if state.return_declared {
                // XXX: the cloning here is bad
                write!(state, "ret = if_then_else(ret_mask & {}(", ret_mask_type);
                show_hir_expr(state, &state.mask.clone().unwrap());
                state.write("), ");
                show_hir_expr(state, e);
                state.write(", ret);\n");
            } else {
                state.write("ret = ");
                show_hir_expr(state, e);
                state.write(";\n");
            }

            show_indent(state);

            if state.return_declared {
                write!(state, "ret_mask &= ~{}(", ret_mask_type);
            } else {
                write!(state, "ret_mask = ~{}(", ret_mask_type);
            }
            show_hir_expr(state, &state.mask.clone().unwrap());
            state.write(");\n");
            state.return_declared = true;
          } else {
            if state.return_declared {
              state.write("ret = if_then_else(ret_mask, ");
              show_hir_expr(state, e);
              state.write(", ret);\n");
            } else {
              state.write("return ");
              show_hir_expr(state, e);
              state.write(";\n");
            }
          }
        } else {
          state.write("return ");
          show_hir_expr(state, e);
          state.write(";\n");
        }
      } else {
        if state.output_cxx {
          if use_return_mask(state) {
            show_indent(state);
            let ret_mask_type = if state.return_vector { "I32" } else { "int32_t" };
            if state.return_declared {
                write!(state, "ret_mask &= ~{}(", ret_mask_type);
            } else {
                write!(state, "ret_mask = ~{}(", ret_mask_type);
            }
            show_hir_expr(state, &state.mask.clone().unwrap());
            state.write(");\n");
            state.return_declared = true;
          } else {
            state.write("return;\n");
          }
        } else {
          state.write("return;\n");
        }
      }
    }
  }
}

pub fn show_path(state: &OutputState, path: &syntax::Path)
{
  match path {
    syntax::Path::Absolute(s) => {
      let _ = write!(state, "<{}>", s);
    }
    syntax::Path::Relative(s) => {
      let _ = write!(state, "\"{}\"", s);
    }
  }
}

pub fn show_preprocessor(state: &OutputState, pp: &syntax::Preprocessor)
{
  match *pp {
    syntax::Preprocessor::Define(ref pd) => show_preprocessor_define(state, pd),
    syntax::Preprocessor::Else => show_preprocessor_else(state),
    syntax::Preprocessor::ElseIf(ref pei) => show_preprocessor_elseif(state, pei),
    syntax::Preprocessor::EndIf => show_preprocessor_endif(state),
    syntax::Preprocessor::Error(ref pe) => show_preprocessor_error(state, pe),
    syntax::Preprocessor::If(ref pi) => show_preprocessor_if(state, pi),
    syntax::Preprocessor::IfDef(ref pid) => show_preprocessor_ifdef(state, pid),
    syntax::Preprocessor::IfNDef(ref pind) => show_preprocessor_ifndef(state, pind),
    syntax::Preprocessor::Include(ref pi) => show_preprocessor_include(state, pi),
    syntax::Preprocessor::Line(ref pl) => show_preprocessor_line(state, pl),
    syntax::Preprocessor::Pragma(ref pp) => show_preprocessor_pragma(state, pp),
    syntax::Preprocessor::Undef(ref pu) => show_preprocessor_undef(state, pu),
    syntax::Preprocessor::Version(ref pv) => show_preprocessor_version(state, pv),
    syntax::Preprocessor::Extension(ref pe) => show_preprocessor_extension(state, pe),
  }
}

pub fn show_preprocessor_define(state: &OutputState, pd: &syntax::PreprocessorDefine)
{
  match *pd {
    syntax::PreprocessorDefine::ObjectLike {
      ref ident,
      ref value,
    } => {
      let _ = write!(state, "#define {} {}\n", ident, value);
    }

    syntax::PreprocessorDefine::FunctionLike {
      ref ident,
      ref args,
      ref value,
    } => {
      let _ = write!(state, "#define {}(", ident);

      if !args.is_empty() {
        let _ = write!(state, "{}", &args[0]);

        for arg in &args[1..args.len()] {
          let _ = write!(state, ", {}", arg);
        }
      }

      let _ = write!(state, ") {}\n", value);
    }
  }
}

pub fn show_preprocessor_else(state: &OutputState)
{
  state.write("#else\n");
}

pub fn show_preprocessor_elseif(state: &OutputState, pei: &syntax::PreprocessorElseIf)
{
  let _ = write!(state, "#elseif {}\n", pei.condition);
}

pub fn show_preprocessor_error(state: &OutputState, pe: &syntax::PreprocessorError)
{
  let _ = writeln!(state, "#error {}", pe.message);
}

pub fn show_preprocessor_endif(state: &OutputState)
{
  state.write("#endif\n");
}

pub fn show_preprocessor_if(state: &OutputState, pi: &syntax::PreprocessorIf)
{
  let _ = write!(state, "#if {}\n", pi.condition);
}

pub fn show_preprocessor_ifdef(state: &OutputState, pid: &syntax::PreprocessorIfDef)
{
  state.write("#ifdef ");
  show_identifier(state, &pid.ident);
  state.write("\n");
}

pub fn show_preprocessor_ifndef(state: &OutputState, pind: &syntax::PreprocessorIfNDef)
{
  state.write("#ifndef ");
  show_identifier(state, &pind.ident);
  state.write("\n");
}

pub fn show_preprocessor_include(state: &OutputState, pi: &syntax::PreprocessorInclude)
{
  state.write("#include ");
  show_path(state, &pi.path);
  state.write("\n");
}

pub fn show_preprocessor_line(state: &OutputState, pl: &syntax::PreprocessorLine)
{
  let _ = write!(state, "#line {}", pl.line);
  if let Some(source_string_number) = pl.source_string_number {
    let _ = write!(state, " {}", source_string_number);
  }
  state.write("\n");
}

pub fn show_preprocessor_pragma(state: &OutputState, pp: &syntax::PreprocessorPragma)
{
  let _ = writeln!(state, "#pragma {}", pp.command);
}

pub fn show_preprocessor_undef(state: &OutputState, pud: &syntax::PreprocessorUndef)
{
  state.write("#undef ");
  show_identifier(state, &pud.name);
  state.write("\n");
}

pub fn show_preprocessor_version(state: &OutputState, pv: &syntax::PreprocessorVersion)
{
  let _ = write!(state, "#version {}", pv.version);

  if let Some(ref profile) = pv.profile {
    match *profile {
      syntax::PreprocessorVersionProfile::Core => {
        state.write(" core");
      }
      syntax::PreprocessorVersionProfile::Compatibility => {
        state.write(" compatibility");
      }
      syntax::PreprocessorVersionProfile::ES => {
        state.write(" es");
      }
    }
  }

  state.write("\n");
}

pub fn show_preprocessor_extension(state: &OutputState, pe: &syntax::PreprocessorExtension)
{
  state.write("#extension ");

  match pe.name {
    syntax::PreprocessorExtensionName::All => {
      state.write("all");
    }
    syntax::PreprocessorExtensionName::Specific(ref n) => {
      state.write(n);
    }
  }

  if let Some(ref behavior) = pe.behavior {
    match *behavior {
      syntax::PreprocessorExtensionBehavior::Require => {
        state.write(" : require");
      }
      syntax::PreprocessorExtensionBehavior::Enable => {
        state.write(" : enable");
      }
      syntax::PreprocessorExtensionBehavior::Warn => {
        state.write(" : warn");
      }
      syntax::PreprocessorExtensionBehavior::Disable => {
        state.write(" : disable");
      }
    }
  }

  state.write("\n");
}


pub fn show_external_declaration(state: &mut OutputState, ed: &hir::ExternalDeclaration) {
  match *ed {
    hir::ExternalDeclaration::Preprocessor(ref pp) => {
      if !state.output_cxx { show_preprocessor(state, pp) }
    }
    hir::ExternalDeclaration::FunctionDefinition(ref fd) => {
      if !state.output_cxx { show_function_definition(state, fd, !0, hir::RunClass::Unknown) }
    }
    hir::ExternalDeclaration::Declaration(ref d) => show_declaration(state, d)
  }
}

pub fn show_cxx_function_definition(state: &mut OutputState, name: hir::SymRef, vector_mask: u32) {
  if let Some((ref fd, run_class)) = state.hir.function_definition(name) {
    state.vector_mask = vector_mask;
    state.return_vector = (vector_mask & (1 << 31)) != 0 || match run_class {
       hir::RunClass::Scalar => false,
       hir::RunClass::Dependent(mask) => (mask & vector_mask) != 0,
       _ => true,
    };
    match state.functions.get(&(name, vector_mask)) {
      Some(true) => {}
      Some(false) => {
        show_function_prototype(state, &fd.prototype);
        state.functions.insert((name, vector_mask), true);
      }
      None => {
        state.functions.insert((name, vector_mask), false);
        let buffer = state.push_buffer();
        show_function_definition(state, fd, vector_mask, run_class);
        for (name, vector_mask) in state.deps.replace(Vec::new()) {
            show_cxx_function_definition(state, name, vector_mask);
        }
        state.flush_buffer();
        state.pop_buffer(buffer);
        state.functions.insert((name, vector_mask), true);
      }
    }
  }
}

pub fn show_translation_unit(state: &mut OutputState, tu: &hir::TranslationUnit) {
  state.flush_buffer();

  for ed in &(tu.0).0 {
    show_external_declaration(state, ed);
    state.flush_buffer();
  }
  if state.output_cxx {
    if let Some(name) = state.hir.lookup("main") {
      show_cxx_function_definition(state, name, 0);
      state.flush_buffer();
    }
  }
}

fn write_abi(state: &mut OutputState, name: &str) {
    match state.kind {
      ShaderKind::Fragment => {
        if state.uses_discard {
            state.write("bool use_discard() { return true; }\n");
        } else {
            state.write("bool use_discard() { return false; }\n");
        }
        state.write("void run(const void* step_ptr) {\n");
        if (state.uses_discard) {
            state.write(" isPixelDiscarded = false;\n");
        }
        state.write(" main();\n");
        state.write(" step_interp_inputs(step_ptr);\n");
        state.write("}\n");
        state.write("void skip(const void* step_ptr) {\n");
        state.write(" step_interp_inputs(step_ptr);\n");
        state.write("}\n");
      }
      ShaderKind::Vertex => {
        state.write("void run(char* flats, char* interps, size_t interp_stride) {\n");
        state.write(" main();\n");
        state.write(" store_flat_outputs(flats);\n");
        state.write(" store_interp_outputs(interps, interp_stride);\n");
        state.write("}\n");
      }
    }
    state.write(name);
    state.write("() {\n");
    write!(state, " typedef {} Self;\n", name);
    state.write(" set_uniform_1i_func = (SetUniform1iFunc)&Self::set_uniform_1i;\n");
    state.write(" set_uniform_4fv_func = (SetUniform4fvFunc)&Self::set_uniform_4fv;\n");
    state.write(" set_uniform_matrix4fv_func = (SetUniformMatrix4fvFunc)&Self::set_uniform_matrix4fv;\n");
    match state.kind {
      ShaderKind::Fragment => {
        state.write(" init_batch_func = (InitBatchFunc)&Self::bind_textures;\n");
        state.write(" init_primitive_func = (InitPrimitiveFunc)&Self::read_flat_inputs;\n");
        state.write(" init_span_func = (InitSpanFunc)&Self::read_interp_inputs;\n");
        state.write(" run_func = (RunFunc)&Self::run;\n");
        state.write(" skip_func = (SkipFunc)&Self::skip;\n");
        state.write(" use_discard_func = (UseDiscardFunc)&Self::use_discard;\n");
      }
      ShaderKind::Vertex => {
        state.write(" get_uniform_func = (GetUniformFunc)&Self::get_uniform_location;\n");
        state.write(" bind_attrib_func = (BindAttribFunc)&Self::bind_attrib_location;\n");
        state.write(" init_batch_func = (InitBatchFunc)&Self::bind_textures;\n");
        state.write(" load_attribs_func = (LoadAttribsFunc)&Self::load_attribs;\n");
        state.write(" run_func = (RunFunc)&Self::run;\n");
      }
    }
    state.write("}\n");
}

