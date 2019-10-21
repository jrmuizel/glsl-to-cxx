extern crate glsl;

use glsl::parser::Parse;
use glsl::syntax::{TranslationUnit, UnaryOp};

mod hir;

use hir::State;
use hir::Type;
use std::io::Read;

fn main() {

  let file = std::env::args().nth(1).unwrap();

  let mut contents = String::new();
  std::fs::File::open(file).unwrap().read_to_string(&mut contents).unwrap();
  let r = TranslationUnit::parse(contents);

  //println!("{:#?}", r);
  let mut output_glsl = String::new();

  let mut ast_glsl = String::new();
  let r = r.unwrap();
  glsl::transpiler::glsl::show_translation_unit(&mut ast_glsl, &r);

  let mut state = hir::State::new();
  let hir = hir::ast_to_hir(&mut state, &r);
  //println!("{:#?}", state);

  let mut uniforms = Vec::new();
  let mut inputs = Vec::new();

  for i in &hir {
    match i {
      hir::ExternalDeclaration::Declaration(hir::Declaration::InitDeclaratorList(ref d))  => {
        match &state.sym(d.head.name).decl {
          hir::SymDecl::Variable(storage, _) => {
            match storage {
              hir::StorageClass::Uniform => {
                uniforms.push(d.head.name);
              }
              hir::StorageClass::In => {
                inputs.push(d.head.name)
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

  let mut state = OutputState { hir: state, indent: 0,
    should_indent: false,
    output_cxx: false,
    in_loop_declaration: false,
    mask: None,
    return_type: None,
    return_declared: false,
    flat: false
  };




  show_translation_unit(&mut output_glsl, &mut state, &hir);

  state.should_indent = true;
  state.output_cxx = true;
  let mut output_cxx = String::new();

  write!(&mut output_cxx, "/* uniforms\n");
  for i in uniforms {
    show_variable(&mut output_cxx, &mut state, &i);
    write!(&mut output_cxx, "\n");
  }
  write!(&mut output_cxx, "*/\n");

  write!(&mut output_cxx, "/* inputs\n");
  for i in inputs {
    show_variable(&mut output_cxx, &mut state, &i);
    write!(&mut output_cxx, "\n");
  }
  write!(&mut output_cxx, "*/\n");

  show_translation_unit(&mut output_cxx, &mut state, &hir);
  use std::io::Write;
  let mut fast = std::fs::File::create("ast").unwrap();
  fast.write(ast_glsl.as_bytes());
  let mut hir = std::fs::File::create("hir").unwrap();
  hir.write(output_glsl.as_bytes());


  println!("{}", output_cxx);
}

pub struct OutputState {
  hir: hir::State,
  should_indent: bool,
  output_cxx: bool,
  indent: i32,
  in_loop_declaration: bool,
  mask: Option<Box<hir::Expr>>,
  return_type: Option<Box<hir::Type>>,
  return_declared: bool,
  flat: bool
}

impl OutputState {
  fn indent(&mut self) {
    if self.should_indent { self.indent += 1 }
  }
  fn outdent(&mut self) {
    if self.should_indent { self.indent -= 1 }
  }
}

use std::fmt::Write;

use glsl::syntax;
use crate::hir::{SwitchStatement, SwizzleSelector};

pub fn show_identifier<F>(f: &mut F, i: &syntax::Identifier) where F: Write {
  let _ = f.write_str(&i.0);
}

pub fn show_sym<F>(f: &mut F, state: &OutputState, i: &hir::SymRef) where F: Write {
  let sym = state.hir.sym(*i);
  match &sym.decl {
    hir::SymDecl::Variable(..) | hir::SymDecl::Function(..) | hir::SymDecl::Struct(..) => {
      let mut name = sym.name.as_str();
      // we want to replace constructor names
      if state.output_cxx {
        name = match name {
          "int" => { "I32" }
          _ => { name }
        };
      }
      let _ = f.write_str(name);
    }
    _ => panic!()
  }
}

pub fn show_variable<F>(f: &mut F, state: &OutputState, i: &hir::SymRef) where F: Write {
  let sym = state.hir.sym(*i);
  match &sym.decl {
    hir::SymDecl::Variable(_, ty) => {
      show_type(f, state, ty);
      let _ = f.write_str(" ");
      let mut name = sym.name.as_str();
      if state.output_cxx {
        name = match name {
          "int" => { "I32" }
          _ => { name }
        };
      }
      let _ = f.write_str(name);
    }
    _ => panic!()
  }
}

pub fn write_constructor<F>(f: &mut F, state: &OutputState, name: &str, s: &hir::StructFields) where F: Write {

  // write default constructor
  let _ = write!(f, "{}() = default;\n", name);

  let _ = write!(f, "{}(", name);
  let mut first_field = true;
  for field in &s.fields {
    if !first_field {
      let _ = f.write_str(", ");
    }

    show_type(f, state, &field.ty);

    let _ = f.write_str(" ");

    show_arrayed_identifier(f, state, &field.name, &field.ty);
    first_field = false;
  }
  let _ = f.write_str(") : ");

  let mut first_field = true;
  for field in &s.fields {
    if !first_field {
      let _ = f.write_str(", ");
    }
    let _ = write!(f, "{}({})", field.name.as_str(), field.name);
    first_field = false;
  }
  let _ = f.write_str("{}\n");
}

pub fn show_storage_class<F>(f: &mut F, q: &hir::StorageClass) where F: Write {
  match *q {
    hir::StorageClass::None => {  }
    hir::StorageClass::Const => { let _ = f.write_str("const "); }
    hir::StorageClass::In => { let _ = f.write_str("in "); }
    hir::StorageClass::Out => { let _ = f.write_str("out "); }
    hir::StorageClass::Uniform => { let _ = f.write_str("uniform "); }
  }
}

pub fn show_sym_decl<F>(f: &mut F, state: &OutputState, i: &hir::SymRef) where F: Write {
  let sym = state.hir.sym(*i);
  match &sym.decl {
    hir::SymDecl::Variable(storage, ..) => {
      if !state.output_cxx {
        show_storage_class(f, storage)
      }
      let mut name = sym.name.as_str();
      if state.output_cxx {
        name = match name {
          "int" => { "I32" }
          _ => { name }
        };
      }
      let _ = f.write_str(name);
    }
    hir::SymDecl::Struct(s) => {
      let _ = f.write_str("struct ");

      let mut name = sym.name.as_str();
      let _ = write!(f, "{} ", name);

      let _ = f.write_str("{\n");

      for field in &s.fields {
        show_struct_field(f, state, field);
      }

      // write if_then_else
      if state.output_cxx {
        write_constructor(f, state, name, s);

        let _ = write!(f, "friend {} if_then_else(I32 c, {} t, {} e) {{\n", name, name, name);
        let _ = write!(f, "return {}(\n", name);
        let mut first_field = true;
        for field in &s.fields {
          if !first_field {
            let _ = f.write_str(", ");
          }
          let _ = write!(f, "if_then_else(c, t.{}, e.{})", field.name, field.name);
          first_field = false;
        }
        let _ = f.write_str(");\n}");

      }
      let _ = f.write_str("}");

    }
    _ => panic!()
  }
}

pub fn show_type_name<F>(f: &mut F, t: &syntax::TypeName) where F: Write {
  let _ = f.write_str(&t.0);
}

pub fn show_type_specifier_non_array<F>(f: &mut F, state: &mut OutputState, t: &syntax::TypeSpecifierNonArray) where F: Write {
  match *t {
    syntax::TypeSpecifierNonArray::Void => { let _ = f.write_str("void"); }
    syntax::TypeSpecifierNonArray::Bool => {
      if state.output_cxx {
        let _ = f.write_str("Bool");
      } else {
        let _ = f.write_str("bool");
      }
    }
    syntax::TypeSpecifierNonArray::Int => {
      if state.output_cxx {
        if state.in_loop_declaration || state.flat {
          let _ = f.write_str("int");
        } else {
          let _ = f.write_str("I32");
        }
      } else {
        let _ = f.write_str("int");
      }
    }
    syntax::TypeSpecifierNonArray::UInt => { let _ = f.write_str("uint"); }
    syntax::TypeSpecifierNonArray::Float => { if state.output_cxx { let _ = f.write_str("Float"); } else { let _ = f.write_str("float"); } }
    syntax::TypeSpecifierNonArray::Double => { let _ = f.write_str("double"); }
    syntax::TypeSpecifierNonArray::Vec2 => { let _ = f.write_str("vec2"); }
    syntax::TypeSpecifierNonArray::Vec3 => { let _ = f.write_str("vec3"); }
    syntax::TypeSpecifierNonArray::Vec4 => { let _ = f.write_str("vec4"); }
    syntax::TypeSpecifierNonArray::DVec2 => { let _ = f.write_str("dvec2"); }
    syntax::TypeSpecifierNonArray::DVec3 => { let _ = f.write_str("dvec3"); }
    syntax::TypeSpecifierNonArray::DVec4 => { let _ = f.write_str("dvec4"); }
    syntax::TypeSpecifierNonArray::BVec2 => { let _ = f.write_str("bvec2"); }
    syntax::TypeSpecifierNonArray::BVec3 => { let _ = f.write_str("bvec3"); }
    syntax::TypeSpecifierNonArray::BVec4 => { let _ = f.write_str("bvec4"); }
    syntax::TypeSpecifierNonArray::IVec2 => { let _ = f.write_str("ivec2"); }
    syntax::TypeSpecifierNonArray::IVec3 => { let _ = f.write_str("ivec3"); }
    syntax::TypeSpecifierNonArray::IVec4 => { let _ = f.write_str("ivec4"); }
    syntax::TypeSpecifierNonArray::UVec2 => { let _ = f.write_str("uvec2"); }
    syntax::TypeSpecifierNonArray::UVec3 => { let _ = f.write_str("uvec3"); }
    syntax::TypeSpecifierNonArray::UVec4 => { let _ = f.write_str("uvec4"); }
    syntax::TypeSpecifierNonArray::Mat2 => { let _ = f.write_str("mat2"); }
    syntax::TypeSpecifierNonArray::Mat3 => { let _ = f.write_str("mat3"); }
    syntax::TypeSpecifierNonArray::Mat4 => { let _ = f.write_str("mat4"); }
    syntax::TypeSpecifierNonArray::Mat23 => { let _ = f.write_str("mat23"); }
    syntax::TypeSpecifierNonArray::Mat24 => { let _ = f.write_str("mat24"); }
    syntax::TypeSpecifierNonArray::Mat32 => { let _ = f.write_str("mat32"); }
    syntax::TypeSpecifierNonArray::Mat34 => { let _ = f.write_str("mat34"); }
    syntax::TypeSpecifierNonArray::Mat42 => { let _ = f.write_str("mat42"); }
    syntax::TypeSpecifierNonArray::Mat43 => { let _ = f.write_str("mat43"); }
    syntax::TypeSpecifierNonArray::DMat2 => { let _ = f.write_str("dmat2"); }
    syntax::TypeSpecifierNonArray::DMat3 => { let _ = f.write_str("dmat3"); }
    syntax::TypeSpecifierNonArray::DMat4 => { let _ = f.write_str("dmat4"); }
    syntax::TypeSpecifierNonArray::DMat23 => { let _ = f.write_str("dmat23"); }
    syntax::TypeSpecifierNonArray::DMat24 => { let _ = f.write_str("dmat24"); }
    syntax::TypeSpecifierNonArray::DMat32 => { let _ = f.write_str("dmat32"); }
    syntax::TypeSpecifierNonArray::DMat34 => { let _ = f.write_str("dmat34"); }
    syntax::TypeSpecifierNonArray::DMat42 => { let _ = f.write_str("dmat42"); }
    syntax::TypeSpecifierNonArray::DMat43 => { let _ = f.write_str("dmat43"); }
    syntax::TypeSpecifierNonArray::Sampler1D => { let _ = f.write_str("sampler1D"); }
    syntax::TypeSpecifierNonArray::Image1D => { let _ = f.write_str("image1D"); }
    syntax::TypeSpecifierNonArray::Sampler2D => { let _ = f.write_str("sampler2D"); }
    syntax::TypeSpecifierNonArray::Image2D => { let _ = f.write_str("image2D"); }
    syntax::TypeSpecifierNonArray::Sampler3D => { let _ = f.write_str("sampler3D"); }
    syntax::TypeSpecifierNonArray::Image3D => { let _ = f.write_str("image3D"); }
    syntax::TypeSpecifierNonArray::SamplerCube => { let _ = f.write_str("samplerCube"); }
    syntax::TypeSpecifierNonArray::ImageCube => { let _ = f.write_str("imageCube"); }
    syntax::TypeSpecifierNonArray::Sampler2DRect => { let _ = f.write_str("sampler2DRect"); }
    syntax::TypeSpecifierNonArray::Image2DRect => { let _ = f.write_str("image2DRect"); }
    syntax::TypeSpecifierNonArray::Sampler1DArray => { let _ = f.write_str("sampler1DArray"); }
    syntax::TypeSpecifierNonArray::Image1DArray => { let _ = f.write_str("image1DArray"); }
    syntax::TypeSpecifierNonArray::Sampler2DArray => { let _ = f.write_str("sampler2DArray"); }
    syntax::TypeSpecifierNonArray::Image2DArray => { let _ = f.write_str("image2DArray"); }
    syntax::TypeSpecifierNonArray::SamplerBuffer => { let _ = f.write_str("samplerBuffer"); }
    syntax::TypeSpecifierNonArray::ImageBuffer => { let _ = f.write_str("imageBuffer"); }
    syntax::TypeSpecifierNonArray::Sampler2DMS => { let _ = f.write_str("sampler2DMS"); }
    syntax::TypeSpecifierNonArray::Image2DMS => { let _ = f.write_str("image2DMS"); }
    syntax::TypeSpecifierNonArray::Sampler2DMSArray => { let _ = f.write_str("sampler2DMSArray"); }
    syntax::TypeSpecifierNonArray::Image2DMSArray => { let _ = f.write_str("image2DMSArray"); }
    syntax::TypeSpecifierNonArray::SamplerCubeArray => { let _ = f.write_str("samplerCubeArray"); }
    syntax::TypeSpecifierNonArray::ImageCubeArray => { let _ = f.write_str("imageCubeArray"); }
    syntax::TypeSpecifierNonArray::Sampler1DShadow => { let _ = f.write_str("sampler1DShadow"); }
    syntax::TypeSpecifierNonArray::Sampler2DShadow => { let _ = f.write_str("sampler2DShadow"); }
    syntax::TypeSpecifierNonArray::Sampler2DRectShadow => { let _ = f.write_str("sampler2DRectShadow"); }
    syntax::TypeSpecifierNonArray::Sampler1DArrayShadow => { let _ = f.write_str("sampler1DArrayShadow"); }
    syntax::TypeSpecifierNonArray::Sampler2DArrayShadow => { let _ = f.write_str("sampler2DArrayShadow"); }
    syntax::TypeSpecifierNonArray::SamplerCubeShadow => { let _ = f.write_str("samplerCubeShadow"); }
    syntax::TypeSpecifierNonArray::SamplerCubeArrayShadow => { let _ = f.write_str("samplerCubeArrayShadow"); }
    syntax::TypeSpecifierNonArray::ISampler1D => { let _ = f.write_str("isampler1D"); }
    syntax::TypeSpecifierNonArray::IImage1D => { let _ = f.write_str("iimage1D"); }
    syntax::TypeSpecifierNonArray::ISampler2D => { let _ = f.write_str("isampler2D"); }
    syntax::TypeSpecifierNonArray::IImage2D => { let _ = f.write_str("iimage2D"); }
    syntax::TypeSpecifierNonArray::ISampler3D => { let _ = f.write_str("isampler3D"); }
    syntax::TypeSpecifierNonArray::IImage3D => { let _ = f.write_str("iimage3D"); }
    syntax::TypeSpecifierNonArray::ISamplerCube => { let _ = f.write_str("isamplerCube"); }
    syntax::TypeSpecifierNonArray::IImageCube => { let _ = f.write_str("iimageCube"); }
    syntax::TypeSpecifierNonArray::ISampler2DRect => { let _ = f.write_str("isampler2DRect"); }
    syntax::TypeSpecifierNonArray::IImage2DRect => { let _ = f.write_str("iimage2DRect"); }
    syntax::TypeSpecifierNonArray::ISampler1DArray => { let _ = f.write_str("isampler1DArray"); }
    syntax::TypeSpecifierNonArray::IImage1DArray => { let _ = f.write_str("iimage1DArray"); }
    syntax::TypeSpecifierNonArray::ISampler2DArray => { let _ = f.write_str("isampler2DArray"); }
    syntax::TypeSpecifierNonArray::IImage2DArray => { let _ = f.write_str("iimage2DArray"); }
    syntax::TypeSpecifierNonArray::ISamplerBuffer => { let _ = f.write_str("isamplerBuffer"); }
    syntax::TypeSpecifierNonArray::IImageBuffer => { let _ = f.write_str("iimageBuffer"); }
    syntax::TypeSpecifierNonArray::ISampler2DMS => { let _ = f.write_str("isampler2MS"); }
    syntax::TypeSpecifierNonArray::IImage2DMS => { let _ = f.write_str("iimage2DMS"); }
    syntax::TypeSpecifierNonArray::ISampler2DMSArray => { let _ = f.write_str("isampler2DMSArray"); }
    syntax::TypeSpecifierNonArray::IImage2DMSArray => { let _ = f.write_str("iimage2DMSArray"); }
    syntax::TypeSpecifierNonArray::ISamplerCubeArray => { let _ = f.write_str("isamplerCubeArray"); }
    syntax::TypeSpecifierNonArray::IImageCubeArray => { let _ = f.write_str("iimageCubeArray"); }
    syntax::TypeSpecifierNonArray::AtomicUInt => { let _ = f.write_str("atomic_uint"); }
    syntax::TypeSpecifierNonArray::USampler1D => { let _ = f.write_str("usampler1D"); }
    syntax::TypeSpecifierNonArray::UImage1D => { let _ = f.write_str("uimage1D"); }
    syntax::TypeSpecifierNonArray::USampler2D => { let _ = f.write_str("usampler2D"); }
    syntax::TypeSpecifierNonArray::UImage2D => { let _ = f.write_str("uimage2D"); }
    syntax::TypeSpecifierNonArray::USampler3D => { let _ = f.write_str("usampler3D"); }
    syntax::TypeSpecifierNonArray::UImage3D => { let _ = f.write_str("uimage3D"); }
    syntax::TypeSpecifierNonArray::USamplerCube => { let _ = f.write_str("usamplerCube"); }
    syntax::TypeSpecifierNonArray::UImageCube => { let _ = f.write_str("uimageCube"); }
    syntax::TypeSpecifierNonArray::USampler2DRect => { let _ = f.write_str("usampler2DRect"); }
    syntax::TypeSpecifierNonArray::UImage2DRect => { let _ = f.write_str("uimage2DRect"); }
    syntax::TypeSpecifierNonArray::USampler1DArray => { let _ = f.write_str("usampler1DArray"); }
    syntax::TypeSpecifierNonArray::UImage1DArray => { let _ = f.write_str("uimage1DArray"); }
    syntax::TypeSpecifierNonArray::USampler2DArray => { let _ = f.write_str("usampler2DArray"); }
    syntax::TypeSpecifierNonArray::UImage2DArray => { let _ = f.write_str("uimage2DArray"); }
    syntax::TypeSpecifierNonArray::USamplerBuffer => { let _ = f.write_str("usamplerBuffer"); }
    syntax::TypeSpecifierNonArray::UImageBuffer => { let _ = f.write_str("uimageBuffer"); }
    syntax::TypeSpecifierNonArray::USampler2DMS => { let _ = f.write_str("usampler2DMS"); }
    syntax::TypeSpecifierNonArray::UImage2DMS => { let _ = f.write_str("uimage2DMS"); }
    syntax::TypeSpecifierNonArray::USampler2DMSArray => { let _ = f.write_str("usamplerDMSArray"); }
    syntax::TypeSpecifierNonArray::UImage2DMSArray => { let _ = f.write_str("uimage2DMSArray"); }
    syntax::TypeSpecifierNonArray::USamplerCubeArray => { let _ = f.write_str("usamplerCubeArray"); }
    syntax::TypeSpecifierNonArray::UImageCubeArray => { let _ = f.write_str("uimageCubeArray"); }
    syntax::TypeSpecifierNonArray::Struct(ref s) => panic!(),//show_struct_non_declaration(f, state, s),
    syntax::TypeSpecifierNonArray::TypeName(ref tn) => show_type_name(f, tn)
  }
}

pub fn show_type_kind<F>(f: &mut F, state: &OutputState, t: &hir::TypeKind) where F: Write {
  match *t {
    hir::TypeKind::Void => { let _ = f.write_str("void"); }
    hir::TypeKind::Bool => {
      if state.output_cxx {
        let _ = f.write_str("Bool");
      } else {
        let _ = f.write_str("bool");
      }
    }
    hir::TypeKind::Int => {
      if state.output_cxx {
        if state.in_loop_declaration || state.flat {
          let _ = f.write_str("int");
        } else {
          let _ = f.write_str("I32");
        }
      } else {
        let _ = f.write_str("int");
      }
    }
    hir::TypeKind::UInt => { let _ = f.write_str("uint"); }
    hir::TypeKind::Float => { if state.output_cxx { let _ = f.write_str("Float"); } else { let _ = f.write_str("float"); } }
    hir::TypeKind::Double => { let _ = f.write_str("double"); }
    hir::TypeKind::Vec2 => { let _ = f.write_str("vec2"); }
    hir::TypeKind::Vec3 => { let _ = f.write_str("vec3"); }
    hir::TypeKind::Vec4 => { let _ = f.write_str("vec4"); }
    hir::TypeKind::DVec2 => { let _ = f.write_str("dvec2"); }
    hir::TypeKind::DVec3 => { let _ = f.write_str("dvec3"); }
    hir::TypeKind::DVec4 => { let _ = f.write_str("dvec4"); }
    hir::TypeKind::BVec2 => { let _ = f.write_str("bvec2"); }
    hir::TypeKind::BVec3 => { let _ = f.write_str("bvec3"); }
    hir::TypeKind::BVec4 => { let _ = f.write_str("bvec4"); }
    hir::TypeKind::IVec2 => { let _ = f.write_str("ivec2"); }
    hir::TypeKind::IVec3 => { let _ = f.write_str("ivec3"); }
    hir::TypeKind::IVec4 => { let _ = f.write_str("ivec4"); }
    hir::TypeKind::UVec2 => { let _ = f.write_str("uvec2"); }
    hir::TypeKind::UVec3 => { let _ = f.write_str("uvec3"); }
    hir::TypeKind::UVec4 => { let _ = f.write_str("uvec4"); }
    hir::TypeKind::Mat2 => { let _ = f.write_str("mat2"); }
    hir::TypeKind::Mat3 => { let _ = f.write_str("mat3"); }
    hir::TypeKind::Mat4 => { let _ = f.write_str("mat4"); }
    hir::TypeKind::Mat23 => { let _ = f.write_str("mat23"); }
    hir::TypeKind::Mat24 => { let _ = f.write_str("mat24"); }
    hir::TypeKind::Mat32 => { let _ = f.write_str("mat32"); }
    hir::TypeKind::Mat34 => { let _ = f.write_str("mat34"); }
    hir::TypeKind::Mat42 => { let _ = f.write_str("mat42"); }
    hir::TypeKind::Mat43 => { let _ = f.write_str("mat43"); }
    hir::TypeKind::DMat2 => { let _ = f.write_str("dmat2"); }
    hir::TypeKind::DMat3 => { let _ = f.write_str("dmat3"); }
    hir::TypeKind::DMat4 => { let _ = f.write_str("dmat4"); }
    hir::TypeKind::DMat23 => { let _ = f.write_str("dmat23"); }
    hir::TypeKind::DMat24 => { let _ = f.write_str("dmat24"); }
    hir::TypeKind::DMat32 => { let _ = f.write_str("dmat32"); }
    hir::TypeKind::DMat34 => { let _ = f.write_str("dmat34"); }
    hir::TypeKind::DMat42 => { let _ = f.write_str("dmat42"); }
    hir::TypeKind::DMat43 => { let _ = f.write_str("dmat43"); }
    hir::TypeKind::Sampler1D => { let _ = f.write_str("sampler1D"); }
    hir::TypeKind::Image1D => { let _ = f.write_str("image1D"); }
    hir::TypeKind::Sampler2D => { let _ = f.write_str("sampler2D"); }
    hir::TypeKind::Image2D => { let _ = f.write_str("image2D"); }
    hir::TypeKind::Sampler3D => { let _ = f.write_str("sampler3D"); }
    hir::TypeKind::Image3D => { let _ = f.write_str("image3D"); }
    hir::TypeKind::SamplerCube => { let _ = f.write_str("samplerCube"); }
    hir::TypeKind::ImageCube => { let _ = f.write_str("imageCube"); }
    hir::TypeKind::Sampler2DRect => { let _ = f.write_str("sampler2DRect"); }
    hir::TypeKind::Image2DRect => { let _ = f.write_str("image2DRect"); }
    hir::TypeKind::Sampler1DArray => { let _ = f.write_str("sampler1DArray"); }
    hir::TypeKind::Image1DArray => { let _ = f.write_str("image1DArray"); }
    hir::TypeKind::Sampler2DArray => { let _ = f.write_str("sampler2DArray"); }
    hir::TypeKind::Image2DArray => { let _ = f.write_str("image2DArray"); }
    hir::TypeKind::SamplerBuffer => { let _ = f.write_str("samplerBuffer"); }
    hir::TypeKind::ImageBuffer => { let _ = f.write_str("imageBuffer"); }
    hir::TypeKind::Sampler2DMS => { let _ = f.write_str("sampler2DMS"); }
    hir::TypeKind::Image2DMS => { let _ = f.write_str("image2DMS"); }
    hir::TypeKind::Sampler2DMSArray => { let _ = f.write_str("sampler2DMSArray"); }
    hir::TypeKind::Image2DMSArray => { let _ = f.write_str("image2DMSArray"); }
    hir::TypeKind::SamplerCubeArray => { let _ = f.write_str("samplerCubeArray"); }
    hir::TypeKind::ImageCubeArray => { let _ = f.write_str("imageCubeArray"); }
    hir::TypeKind::Sampler1DShadow => { let _ = f.write_str("sampler1DShadow"); }
    hir::TypeKind::Sampler2DShadow => { let _ = f.write_str("sampler2DShadow"); }
    hir::TypeKind::Sampler2DRectShadow => { let _ = f.write_str("sampler2DRectShadow"); }
    hir::TypeKind::Sampler1DArrayShadow => { let _ = f.write_str("sampler1DArrayShadow"); }
    hir::TypeKind::Sampler2DArrayShadow => { let _ = f.write_str("sampler2DArrayShadow"); }
    hir::TypeKind::SamplerCubeShadow => { let _ = f.write_str("samplerCubeShadow"); }
    hir::TypeKind::SamplerCubeArrayShadow => { let _ = f.write_str("samplerCubeArrayShadow"); }
    hir::TypeKind::ISampler1D => { let _ = f.write_str("isampler1D"); }
    hir::TypeKind::IImage1D => { let _ = f.write_str("iimage1D"); }
    hir::TypeKind::ISampler2D => { let _ = f.write_str("isampler2D"); }
    hir::TypeKind::IImage2D => { let _ = f.write_str("iimage2D"); }
    hir::TypeKind::ISampler3D => { let _ = f.write_str("isampler3D"); }
    hir::TypeKind::IImage3D => { let _ = f.write_str("iimage3D"); }
    hir::TypeKind::ISamplerCube => { let _ = f.write_str("isamplerCube"); }
    hir::TypeKind::IImageCube => { let _ = f.write_str("iimageCube"); }
    hir::TypeKind::ISampler2DRect => { let _ = f.write_str("isampler2DRect"); }
    hir::TypeKind::IImage2DRect => { let _ = f.write_str("iimage2DRect"); }
    hir::TypeKind::ISampler1DArray => { let _ = f.write_str("isampler1DArray"); }
    hir::TypeKind::IImage1DArray => { let _ = f.write_str("iimage1DArray"); }
    hir::TypeKind::ISampler2DArray => { let _ = f.write_str("isampler2DArray"); }
    hir::TypeKind::IImage2DArray => { let _ = f.write_str("iimage2DArray"); }
    hir::TypeKind::ISamplerBuffer => { let _ = f.write_str("isamplerBuffer"); }
    hir::TypeKind::IImageBuffer => { let _ = f.write_str("iimageBuffer"); }
    hir::TypeKind::ISampler2DMS => { let _ = f.write_str("isampler2MS"); }
    hir::TypeKind::IImage2DMS => { let _ = f.write_str("iimage2DMS"); }
    hir::TypeKind::ISampler2DMSArray => { let _ = f.write_str("isampler2DMSArray"); }
    hir::TypeKind::IImage2DMSArray => { let _ = f.write_str("iimage2DMSArray"); }
    hir::TypeKind::ISamplerCubeArray => { let _ = f.write_str("isamplerCubeArray"); }
    hir::TypeKind::IImageCubeArray => { let _ = f.write_str("iimageCubeArray"); }
    hir::TypeKind::AtomicUInt => { let _ = f.write_str("atomic_uint"); }
    hir::TypeKind::USampler1D => { let _ = f.write_str("usampler1D"); }
    hir::TypeKind::UImage1D => { let _ = f.write_str("uimage1D"); }
    hir::TypeKind::USampler2D => { let _ = f.write_str("usampler2D"); }
    hir::TypeKind::UImage2D => { let _ = f.write_str("uimage2D"); }
    hir::TypeKind::USampler3D => { let _ = f.write_str("usampler3D"); }
    hir::TypeKind::UImage3D => { let _ = f.write_str("uimage3D"); }
    hir::TypeKind::USamplerCube => { let _ = f.write_str("usamplerCube"); }
    hir::TypeKind::UImageCube => { let _ = f.write_str("uimageCube"); }
    hir::TypeKind::USampler2DRect => { let _ = f.write_str("usampler2DRect"); }
    hir::TypeKind::UImage2DRect => { let _ = f.write_str("uimage2DRect"); }
    hir::TypeKind::USampler1DArray => { let _ = f.write_str("usampler1DArray"); }
    hir::TypeKind::UImage1DArray => { let _ = f.write_str("uimage1DArray"); }
    hir::TypeKind::USampler2DArray => { let _ = f.write_str("usampler2DArray"); }
    hir::TypeKind::UImage2DArray => { let _ = f.write_str("uimage2DArray"); }
    hir::TypeKind::USamplerBuffer => { let _ = f.write_str("usamplerBuffer"); }
    hir::TypeKind::UImageBuffer => { let _ = f.write_str("uimageBuffer"); }
    hir::TypeKind::USampler2DMS => { let _ = f.write_str("usampler2DMS"); }
    hir::TypeKind::UImage2DMS => { let _ = f.write_str("uimage2DMS"); }
    hir::TypeKind::USampler2DMSArray => { let _ = f.write_str("usamplerDMSArray"); }
    hir::TypeKind::UImage2DMSArray => { let _ = f.write_str("uimage2DMSArray"); }
    hir::TypeKind::USamplerCubeArray => { let _ = f.write_str("usamplerCubeArray"); }
    hir::TypeKind::UImageCubeArray => { let _ = f.write_str("uimageCubeArray"); }
    hir::TypeKind::Struct(ref s) => { let _ = f.write_str(state.hir.sym(*s).name.as_str()); } ,
  }
}

pub fn show_type_specifier<F>(f: &mut F, state: &mut OutputState, t: &syntax::TypeSpecifier) where F: Write {
  show_type_specifier_non_array(f, state, &t.ty);

  if let Some(ref arr_spec) = t.array_specifier {
    show_array_spec(f, arr_spec);
  }
}

pub fn show_type<F>(f: &mut F, state: &OutputState, t: &Type) where F: Write {
  if !state.output_cxx {
    if let Some(ref precision) = t.precision {
      show_precision_qualifier(f, precision);
      let _ = f.write_str(" ");
    }
  }

  if state.output_cxx {
    if let Some(ref array) = t.array_sizes {
      let _ = f.write_str("std::array<");
      show_type_kind(f, state, &t.kind);
      let size = match &array.sizes[..] {
        [size] => size,
        _ => panic!()
      };
      let _ = f.write_str(",");
      show_hir_expr(f, state, size);
      let _ = f.write_str(">");
    } else {
      show_type_kind(f, state, &t.kind);
    }
  } else {
    show_type_kind(f, state, &t.kind);
  }

  /*if let Some(ref arr_spec) = t.array_sizes {
    panic!();
  }*/
}

/*pub fn show_fully_specified_type<F>(f: &mut F, state: &mut OutputState, t: &FullySpecifiedType) where F: Write {
  state.flat = false;
  if let Some(ref qual) = t.qualifier {
    if !state.output_cxx {
      show_type_qualifier(f, &qual);
    } else {
      state.flat = qual.qualifiers.0.iter().flat_map(|q| match q { syntax::TypeQualifierSpec::Interpolation(Flat) => Some(()), _ => None}).next().is_some();
    }
    let _ = f.write_str(" ");
  }

  show_type_specifier(f, state, &t.ty);
}*/

/*pub fn show_struct_non_declaration<F>(f: &mut F, state: &mut OutputState, s: &syntax::StructSpecifier) where F: Write {
  let _ = f.write_str("struct ");

  if let Some(ref name) = s.name {
    let _ = write!(f, "{} ", name);
  }

  let _ = f.write_str("{\n");

  for field in &s.fields.0 {
    show_struct_field(f, state, field);
  }

  let _ = f.write_str("}");
}*/

pub fn show_struct<F>(f: &mut F, state: &mut OutputState, s: &syntax::StructSpecifier) where F: Write {
  panic!();//show_struct_non_declaration(f, state, s);
  let _ = f.write_str(";\n");
}

pub fn show_struct_field<F>(f: &mut F, state: &OutputState, field: &hir::StructField) where F: Write {
  show_type(f, state, &field.ty);
  let _ = f.write_str(" ");

  show_arrayed_identifier(f, state, &field.name, &field.ty);

  let _ = f.write_str(";\n");
}

pub fn show_array_spec<F>(f: &mut F, a: &syntax::ArraySpecifier) where F: Write {
  match *a {
    syntax::ArraySpecifier::Unsized => { let _ = f.write_str("[]"); }
    syntax::ArraySpecifier::ExplicitlySized(ref e) => {
      let _ = f.write_str("[");
      show_expr(f, &e);
      let _ = f.write_str("]");
    }
  }
}

pub fn show_arrayed_identifier<F>(f: &mut F, state: &OutputState, ident: &syntax::Identifier, ty: &hir::Type) where F: Write {
  let _ = write!(f, "{}", ident);

  if let Some(ref arr_spec) = ty.array_sizes {
    show_array_sizes(f, state, &arr_spec);
  }
}

pub fn show_array_sizes<F>(f: &mut F, state: &OutputState, a: &hir::ArraySizes) where F: Write {
  let _ = f.write_str("[");
  match &a.sizes[..] {
    [a] => show_hir_expr(f, state, a),
    _ => panic!()
  }

  let _ = f.write_str("]");
  /*
  match *a {
    syntax::ArraySpecifier::Unsized => { let _ = f.write_str("[]"); }
    syntax::ArraySpecifier::ExplicitlySized(ref e) => {
      let _ = f.write_str("[");
      show_expr(f, &e);
      let _ = f.write_str("]");
    }
  }*/
}

pub fn show_type_qualifier<F>(f: &mut F, q: &hir::TypeQualifier) where F: Write {
  let mut qualifiers = q.qualifiers.0.iter();
  let first = qualifiers.next().unwrap();

  show_type_qualifier_spec(f, first);

  for qual_spec in qualifiers {
    let _ = f.write_str(" ");
    show_type_qualifier_spec(f, qual_spec)
  }
}

pub fn show_type_qualifier_spec<F>(f: &mut F, q: &hir::TypeQualifierSpec) where F: Write {
  match *q {
    hir::TypeQualifierSpec::Layout(ref l) => show_layout_qualifier(f, &l),
    hir::TypeQualifierSpec::Interpolation(ref i) => show_interpolation_qualifier(f, &i),
    hir::TypeQualifierSpec::Parameter(ref p) => panic!(),
    hir::TypeQualifierSpec::Memory(ref m) => panic!(),
    hir::TypeQualifierSpec::Invariant => { let _ = f.write_str("invariant"); },
    hir::TypeQualifierSpec::Precise => { let _ = f.write_str("precise"); }
  }
}

pub fn show_syntax_storage_qualifier<F>(f: &mut F, q: &syntax::StorageQualifier) where F: Write {
  match *q {
    syntax::StorageQualifier::Const => { let _ = f.write_str("const"); }
    syntax::StorageQualifier::InOut => { let _ = f.write_str("inout"); }
    syntax::StorageQualifier::In => { let _ = f.write_str("in"); }
    syntax::StorageQualifier::Out => { let _ = f.write_str("out"); }
    syntax::StorageQualifier::Centroid => { let _ = f.write_str("centroid"); }
    syntax::StorageQualifier::Patch => { let _ = f.write_str("patch"); }
    syntax::StorageQualifier::Sample => { let _ = f.write_str("sample"); }
    syntax::StorageQualifier::Uniform => { let _ = f.write_str("uniform"); }
    syntax::StorageQualifier::Buffer => { let _ = f.write_str("buffer"); }
    syntax::StorageQualifier::Shared => { let _ = f.write_str("shared"); }
    syntax::StorageQualifier::Coherent => { let _ = f.write_str("coherent"); }
    syntax::StorageQualifier::Volatile => { let _ = f.write_str("volatile"); }
    syntax::StorageQualifier::Restrict => { let _ = f.write_str("restrict"); }
    syntax::StorageQualifier::ReadOnly => { let _ = f.write_str("readonly"); }
    syntax::StorageQualifier::WriteOnly => { let _ = f.write_str("writeonly"); }
    syntax::StorageQualifier::Subroutine(ref n) => show_subroutine(f, &n)
  }
}

pub fn show_subroutine<F>(f: &mut F, types: &Vec<syntax::TypeName>) where F: Write {
  let _ = f.write_str("subroutine");

  if !types.is_empty() {
    let _ = f.write_str("(");

    let mut types_iter = types.iter();
    let first = types_iter.next().unwrap();

    show_type_name(f, first);

    for type_name in types_iter {
      let _ = f.write_str(", ");
      show_type_name(f, type_name);
    }

    let _ = f.write_str(")");
  }
}

pub fn show_layout_qualifier<F>(f: &mut F, l: &syntax::LayoutQualifier) where F: Write {
  let mut qualifiers = l.ids.0.iter();
  let first = qualifiers.next().unwrap();

  let _ = f.write_str("layout (");
  show_layout_qualifier_spec(f, first);

  for qual_spec in qualifiers {
    let _ = f.write_str(", ");
    show_layout_qualifier_spec(f, qual_spec);
  }

  let _ = f.write_str(")");
}

pub fn show_layout_qualifier_spec<F>(f: &mut F, l: &syntax::LayoutQualifierSpec) where F: Write {
  match *l {
    syntax::LayoutQualifierSpec::Identifier(ref i, Some(ref e)) => {
      let _ = write!(f, "{} = ", i);
      show_expr(f, &e);
    }
    syntax::LayoutQualifierSpec::Identifier(ref i, None) => show_identifier(f, &i),
    syntax::LayoutQualifierSpec::Shared => { let _ = f.write_str("shared"); }
  }
}

pub fn show_precision_qualifier<F>(f: &mut F, p: &syntax::PrecisionQualifier) where F: Write {
  match *p {
    syntax::PrecisionQualifier::High => { let _ = f.write_str("highp"); }
    syntax::PrecisionQualifier::Medium => { let _ = f.write_str("mediump"); }
    syntax::PrecisionQualifier::Low => { let _ = f.write_str("low"); }
  }
}

pub fn show_interpolation_qualifier<F>(f: &mut F, i: &syntax::InterpolationQualifier) where F: Write {
  match *i {
    syntax::InterpolationQualifier::Smooth => { let _ = f.write_str("smooth"); }
    syntax::InterpolationQualifier::Flat => { let _ = f.write_str("flat"); }
    syntax::InterpolationQualifier::NoPerspective => { let _ = f.write_str("noperspective"); }
  }
}

pub fn show_parameter_qualifier<F>(f: &mut F, state: &mut OutputState, i: &Option<hir::ParameterQualifier>) where F: Write {
  if let Some(i) = i {
    if state.output_cxx {
      match *i {
        hir::ParameterQualifier::Out => { let _ = f.write_str("&"); }
        hir::ParameterQualifier::InOut => { let _ = f.write_str("&"); }
        _ => {}
      }
    } else {
      match *i {
        hir::ParameterQualifier::Const => { let _ = f.write_str("const"); }
        hir::ParameterQualifier::In => { let _ = f.write_str("in"); }
        hir::ParameterQualifier::Out => { let _ = f.write_str("out"); }
        hir::ParameterQualifier::InOut => { let _ = f.write_str("inout"); }
      }
    }
  }
}


pub fn show_float<F>(f: &mut F, x: f32) where F: Write {
  if x.fract() == 0. {
    let _ = write!(f, "{}.", x);
  } else {
    let _ = write!(f, "{}", x);
  }
}

pub fn show_double<F>(f: &mut F, x: f64) where F: Write {
  if x.fract() == 0. {
    let _ = write!(f, "{}.", x);
  } else {
    let _ = write!(f, "{}", x);
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

pub fn show_hir_expr<F>(f: &mut F, state: &OutputState, expr: &hir::Expr) where F: Write {
  match expr.kind {
    hir::ExprKind::Variable(ref i) => show_sym(f, state, i),
    hir::ExprKind::IntConst(ref x) => { let _ = write!(f, "{}", x); }
    hir::ExprKind::UIntConst(ref x) => { let _ = write!(f, "{}u", x); }
    hir::ExprKind::BoolConst(ref x) => { let _ = write!(f, "{}", x); }
    hir::ExprKind::FloatConst(ref x) => show_float(f, *x),
    hir::ExprKind::DoubleConst(ref x) => show_double(f, *x),
    hir::ExprKind::Unary(ref op, ref e) => {
      show_unary_op(f, &op);
      let _ = f.write_str("(");
      show_hir_expr(f, state, &e);
      let _ = f.write_str(")");
    }
    hir::ExprKind::Binary(ref op, ref l, ref r) => {
      let _ = f.write_str("(");
      show_hir_expr(f, state, &l);
      let _ = f.write_str(")");
      show_binary_op(f, &op);
      let _ = f.write_str("(");
      show_hir_expr(f, state, &r);
      let _ = f.write_str(")");
    }
    hir::ExprKind::Ternary(ref c, ref s, ref e) => {
      if state.output_cxx {
        let _ = f.write_str("if_then_else(");
        show_hir_expr(f, state, &c);
        let _ = f.write_str(", ");
        show_hir_expr(f, state, &s);
        let _ = f.write_str(", ");
        show_hir_expr(f, state, &e);
        let _ = f.write_str(")");
      } else {
        show_hir_expr(f, state, &c);
        let _ = f.write_str(" ? ");
        show_hir_expr(f, state, &s);
        let _ = f.write_str(" : ");
        show_hir_expr(f, state, &e);
      }
    }
    hir::ExprKind::Assignment(ref v, ref op, ref e) => {
      show_hir_expr(f, state, &v);
      let _ = f.write_str(" ");
      show_assignment_op(f, &op);
      let _ = f.write_str(" ");
      if let Some(mask) = &state.mask {
        let _ = f.write_str("if_then_else(");
        show_hir_expr(f, state, mask);
        let _ = f.write_str(",");
        show_hir_expr(f, state, &e);
        let _ = f.write_str(",");
        show_hir_expr(f, state, &v);
        let _ = f.write_str(")");
      } else {
        show_hir_expr(f, state, &e);
      }
    }
    hir::ExprKind::Bracket(ref e, ref indx) => {
      show_hir_expr(f, state, &e);
      let _ = f.write_str("[");
      show_hir_expr(f, state, indx);
      let _ = f.write_str("]");
    }
    hir::ExprKind::FunCall(ref fun, ref args) => {
      let array_constructor = match fun {
        hir::FunIdentifier::Constructor(Type { array_sizes: Some(arr), ..}) => true,
        _ => false
      };

      show_hir_function_identifier(f, state, &fun);
      if array_constructor {
        let _ = f.write_str("{");
      } else {
        let _ = f.write_str("(");
      }

      if !args.is_empty() {
        let mut args_iter = args.iter();
        let first = args_iter.next().unwrap();
        show_hir_expr(f, state, first);

        for e in args_iter {
          let _ = f.write_str(", ");
          show_hir_expr(f, state, e);
        }
      }

      if array_constructor {
        let _ = f.write_str("}");
      } else {
        let _ = f.write_str(")");
      }
    }
    hir::ExprKind::Dot(ref e, ref i) => {
      let _ = f.write_str("(");
      show_hir_expr(f, state, &e);
      let _ = f.write_str(")");
      let _ = f.write_str(".");
      show_identifier(f, i);
    }
    hir::ExprKind::SwizzleSelector(ref e, ref s) => {
      if state.output_cxx {
        let _ = f.write_str("(");
        show_hir_expr(f, state, &e);
        let _ = f.write_str(").sel(");
        let _ = f.write_str(&s.to_args());
        let _ = f.write_str(")");
      } else {
        let _ = f.write_str("(");
        show_hir_expr(f, state, &e);
        let _ = f.write_str(")");
        let _ = f.write_str(".");
        let _ = f.write_str(&s.to_string());
      }
    }
    hir::ExprKind::PostInc(ref e) => {
      show_hir_expr(f, state, &e);
      let _ = f.write_str("++");
    }
    hir::ExprKind::PostDec(ref e) => {
      show_hir_expr(f, state, &e);
      let _ = f.write_str("--");
    }
    hir::ExprKind::Comma(ref a, ref b) => {
      show_hir_expr(f, state, &a);
      let _ = f.write_str(", ");
      show_hir_expr(f, state, &b);
    }
  }
}

pub fn show_expr<F>(f: &mut F, expr: &syntax::Expr) where F: Write {
  match *expr {
    syntax::Expr::Variable(ref i) => show_identifier(f, &i),
    syntax::Expr::IntConst(ref x) => { let _ = write!(f, "{}", x); }
    syntax::Expr::UIntConst(ref x) => { let _ = write!(f, "{}u", x); }
    syntax::Expr::BoolConst(ref x) => { let _ = write!(f, "{}", x); }
    syntax::Expr::FloatConst(ref x) => show_float(f, *x),
    syntax::Expr::DoubleConst(ref x) => show_double(f, *x),
    syntax::Expr::Unary(ref op, ref e) => {
      show_unary_op(f, &op);
      let _ = f.write_str("(");
      show_expr(f, &e);
      let _ = f.write_str(")");
    }
    syntax::Expr::Binary(ref op, ref l, ref r) => {
      let _ = f.write_str("(");
      show_expr(f, &l);
      let _ = f.write_str(")");
      show_binary_op(f, &op);
      let _ = f.write_str("(");
      show_expr(f, &r);
      let _ = f.write_str(")");
    }
    syntax::Expr::Ternary(ref c, ref s, ref e) => {
      show_expr(f, &c);
      let _ = f.write_str(" ? ");
      show_expr(f, &s);
      let _ = f.write_str(" : ");
      show_expr(f, &e);
    }
    syntax::Expr::Assignment(ref v, ref op, ref e) => {
      show_expr(f, &v);
      let _ = f.write_str(" ");
      show_assignment_op(f, &op);
      let _ = f.write_str(" ");
      show_expr(f, &e);
    }
    syntax::Expr::Bracket(ref e, ref a) => {
      show_expr(f, &e);
      show_array_spec(f, &a);
    }
    syntax::Expr::FunCall(ref fun, ref args) => {
      show_function_identifier(f, &fun);
      let _ = f.write_str("(");

      if !args.is_empty() {
        let mut args_iter = args.iter();
        let first = args_iter.next().unwrap();
        show_expr(f, first);

        for e in args_iter {
          let _ = f.write_str(", ");
          show_expr(f, e);
        }
      }

      let _ = f.write_str(")");
    }
    syntax::Expr::Dot(ref e, ref i) => {
      let _ = f.write_str("(");
      show_expr(f, &e);
      let _ = f.write_str(")");
      let _ = f.write_str(".");
      show_identifier(f, &i);
    }
    syntax::Expr::PostInc(ref e) => {
      show_expr(f, &e);
      let _ = f.write_str("++");
    }
    syntax::Expr::PostDec(ref e) => {
      show_expr(f, &e);
      let _ = f.write_str("--");
    }
    syntax::Expr::Comma(ref a, ref b) => {
      show_expr(f, &a);
      let _ = f.write_str(", ");
      show_expr(f, &b);
    }
  }
}

pub fn show_unary_op<F>(f: &mut F, op: &syntax::UnaryOp) where F: Write {
  match *op {
    syntax::UnaryOp::Inc => { let _ = f.write_str("++"); }
    syntax::UnaryOp::Dec => { let _ = f.write_str("--"); }
    syntax::UnaryOp::Add => { let _ = f.write_str("+"); }
    syntax::UnaryOp::Minus => { let _ = f.write_str("-"); }
    syntax::UnaryOp::Not => { let _ = f.write_str("!"); }
    syntax::UnaryOp::Complement => { let _ = f.write_str("~"); }
  }
}

pub fn show_binary_op<F>(f: &mut F, op: &syntax::BinaryOp) where F: Write {
  match *op {
    syntax::BinaryOp::Or => { let _ = f.write_str("||"); }
    syntax::BinaryOp::Xor => { let _ = f.write_str("^^"); }
    syntax::BinaryOp::And => { let _ = f.write_str("&&"); }
    syntax::BinaryOp::BitOr => { let _ = f.write_str("|"); }
    syntax::BinaryOp::BitXor => { let _ = f.write_str("^"); }
    syntax::BinaryOp::BitAnd => { let _ = f.write_str("&"); }
    syntax::BinaryOp::Equal => { let _ = f.write_str("=="); }
    syntax::BinaryOp::NonEqual => { let _ = f.write_str("!="); }
    syntax::BinaryOp::LT => { let _ = f.write_str("<"); }
    syntax::BinaryOp::GT => { let _ = f.write_str(">"); }
    syntax::BinaryOp::LTE => { let _ = f.write_str("<="); }
    syntax::BinaryOp::GTE => { let _ = f.write_str(">="); }
    syntax::BinaryOp::LShift => { let _ = f.write_str("<<"); }
    syntax::BinaryOp::RShift => { let _ = f.write_str(">>"); }
    syntax::BinaryOp::Add => { let _ = f.write_str("+"); }
    syntax::BinaryOp::Sub => { let _ = f.write_str("-"); }
    syntax::BinaryOp::Mult => { let _ = f.write_str("*"); }
    syntax::BinaryOp::Div => { let _ = f.write_str("/"); }
    syntax::BinaryOp::Mod => { let _ = f.write_str("%"); }
  }
}

pub fn show_assignment_op<F>(f: &mut F, op: &syntax::AssignmentOp) where F: Write {
  match *op {
    syntax::AssignmentOp::Equal => { let _ = f.write_str("="); }
    syntax::AssignmentOp::Mult => { let _ = f.write_str("*="); }
    syntax::AssignmentOp::Div => { let _ = f.write_str("/="); }
    syntax::AssignmentOp::Mod => { let _ = f.write_str("%="); }
    syntax::AssignmentOp::Add => { let _ = f.write_str("+="); }
    syntax::AssignmentOp::Sub => { let _ = f.write_str("-="); }
    syntax::AssignmentOp::LShift => { let _ = f.write_str("<<="); }
    syntax::AssignmentOp::RShift => { let _ = f.write_str(">>="); }
    syntax::AssignmentOp::And => { let _ = f.write_str("&="); }
    syntax::AssignmentOp::Xor => { let _ = f.write_str("^="); }
    syntax::AssignmentOp::Or => { let _ = f.write_str("|="); }
  }
}

pub fn show_function_identifier<F>(f: &mut F, i: &syntax::FunIdentifier) where F: Write {
  match *i {
    syntax::FunIdentifier::Identifier(ref n) => show_identifier(f, &n),
    syntax::FunIdentifier::Expr(ref e) => show_expr(f, &*e)
  }
}

pub fn show_hir_function_identifier<F>(f: &mut F, state: &OutputState, i: &hir::FunIdentifier) where F: Write {
  match *i {
    hir::FunIdentifier::Identifier(ref n) => show_sym(f, state, n),
    hir::FunIdentifier::Constructor(ref t) => show_type(f, state, &*t)
  }
}

pub fn show_declaration<F>(f: &mut F, state: &mut OutputState, d: &hir::Declaration) where F: Write {
  show_indent(f, state);
  match *d {
    hir::Declaration::FunctionPrototype(ref proto) => {
      show_function_prototype(f, state, &proto);
      let _ = f.write_str(";\n");
    }
    hir::Declaration::InitDeclaratorList(ref list) => {
      show_init_declarator_list(f, state, &list);
      let _ = f.write_str(";\n");
    }
    hir::Declaration::Precision(ref qual, ref ty) => {
      if !state.output_cxx {
        show_precision_qualifier(f, &qual);
        show_type_specifier(f, state, &ty);
        let _ = f.write_str(";\n");
      }
    }
    hir::Declaration::Block(ref block) => {
      show_block(f, state, &block);
      let _ = f.write_str(";\n");
    }
    hir::Declaration::Global(ref qual, ref identifiers) => {
      show_type_qualifier(f, &qual);

      if !identifiers.is_empty() {
        let mut iter = identifiers.iter();
        let first = iter.next().unwrap();
        show_identifier(f, first);

        for identifier in iter {
          let _ = write!(f, ", {}", identifier);
        }
      }

      let _ = f.write_str(";\n");
    }
    hir::Declaration::StructDefinition(ref sym) => {
      show_sym_decl(f, state, sym);

      let _ = f.write_str(";\n");
    }
  }
}

pub fn show_function_prototype<F>(f: &mut F, state: &mut OutputState, fp: &hir::FunctionPrototype) where F: Write {
  show_type(f, state, &fp.ty);
  let _ = f.write_str(" ");
  show_identifier(f, &fp.name);

  let _ = f.write_str("(");

  if !fp.parameters.is_empty() {
    let mut iter = fp.parameters.iter();
    let first = iter.next().unwrap();
    show_function_parameter_declaration(f, state, first);

    for param in iter {
      let _ = f.write_str(", ");
      show_function_parameter_declaration(f, state, param);
    }
  }

  let _ = f.write_str(")");
}

pub fn show_function_parameter_declaration<F>(f: &mut F, state: &mut OutputState, p: &hir::FunctionParameterDeclaration) where F: Write {
  match *p {
    hir::FunctionParameterDeclaration::Named(ref qual, ref fpd) => {
      if state.output_cxx {
        show_type(f, state, &fpd.ty);
        show_parameter_qualifier(f, state, qual);
      } else {
        show_parameter_qualifier(f, state, qual);
        let _ = f.write_str(" ");
        show_type(f, state, &fpd.ty);
      }
      let _ = f.write_str(" ");
      show_arrayed_identifier(f, state, &fpd.ident, &fpd.ty);
    }
    hir::FunctionParameterDeclaration::Unnamed(ref qual, ref ty) => {
      if state.output_cxx {
        show_type_specifier(f, state, ty);
        show_parameter_qualifier(f, state, qual);
      } else {
        show_parameter_qualifier(f, state, qual);
        let _ = f.write_str(" ");
        show_type_specifier(f, state, ty);
      }
    }
  }
}

pub fn show_init_declarator_list<F>(f: &mut F, state: &mut OutputState, i: &hir::InitDeclaratorList) where F: Write {
  show_single_declaration(f, state, &i.head);

  for decl in &i.tail {
    let _ = f.write_str(", ");
    show_single_declaration_no_type(f, state, decl);
  }
}

pub fn show_single_declaration<F>(f: &mut F, state: &mut OutputState, d: &hir::SingleDeclaration) where F: Write {
  if state.output_cxx {
    show_single_declaration_cxx(f, state, d)
  } else {
    show_single_declaration_glsl(f, state, d)
  }
}


pub fn show_single_declaration_glsl<F>(f: &mut F, state: &mut OutputState, d: &hir::SingleDeclaration) where F: Write {
  if let Some(ref qual) = d.qualifier {
    show_type_qualifier(f, &qual);
    let _ = f.write_str(" ");
  }

  let sym = state.hir.sym(d.name);
  match &sym.decl {
    hir::SymDecl::Variable(storage, ..) => {
      if !state.output_cxx {
        show_storage_class(f, storage)
      }
    }
    _ => panic!("should be variable")
  }

  if let Some(ty_def) = d.ty_def {
    show_sym_decl(f, state, &ty_def);
  } else {
    show_type(f, state, &d.ty);
  }

  let _ = f.write_str(" ");
  let mut name = sym.name.as_str();
  let _ = f.write_str(name);

  if let Some(ref arr_spec) = d.ty.array_sizes {
    show_array_sizes(f, state, &arr_spec);
  }

  if let Some(ref initializer) = d.initializer {
    let _ = f.write_str(" = ");
    show_initializer(f, state, initializer);
  }

}

pub fn show_single_declaration_cxx<F>(f: &mut F, state: &mut OutputState, d: &hir::SingleDeclaration) where F: Write {
  state.flat = false;
  if let Some(ref qual) = d.qualifier {
    state.flat = qual.qualifiers.0.iter().flat_map(|q| match q { hir::TypeQualifierSpec::Interpolation(Flat) => Some(()), _ => None}).next().is_some();
  }

  if let Some(ref array) = d.ty.array_sizes {
    show_type(f, state, &d.ty);
  } else {
    if let Some(ty_def) = d.ty_def {
      show_sym_decl(f, state, &ty_def);
    } else {
      show_type(f, state, &d.ty);
    }
  }

  // XXX: this is pretty grotty
  let _ = f.write_str(" ");
  show_sym_decl(f, state, &d.name);

  if let Some(ref initializer) = d.initializer {
    let _ = f.write_str(" = ");
    show_initializer(f, state, initializer);
  }

}

pub fn show_single_declaration_no_type<F>(f: &mut F, state: &mut OutputState, d: &hir::SingleDeclarationNoType) where F: Write {
  panic!();//show_arrayed_identifier(f, &d.ident);

  if let Some(ref initializer) = d.initializer {
    let _ = f.write_str(" = ");
    show_initializer(f, state, initializer);
  }
}

pub fn show_initializer<F>(f: &mut F, state: &mut OutputState, i: &hir::Initializer) where F: Write {
  match *i {
    hir::Initializer::Simple(ref e) => show_hir_expr(f, state, e),
    hir::Initializer::List(ref list) => {
      let mut iter = list.0.iter();
      let first = iter.next().unwrap();

      let _ = f.write_str("{ ");
      show_initializer(f, state, first);

      for ini in iter {
        let _ = f.write_str(", ");
        show_initializer(f, state, ini);
      }

      let _ = f.write_str(" }");
    }
  }
}

pub fn show_block<F>(f: &mut F, state: &mut OutputState, b: &hir::Block) where F: Write {
  show_type_qualifier(f, &b.qualifier);
  let _ = f.write_str(" ");
  show_identifier(f, &b.name);
  let _ = f.write_str(" {");

  for field in &b.fields {
    panic!();//show_struct_field(f, state, field);
    let _ = f.write_str("\n");
  }
  let _ = f.write_str("}");

  if let Some(ref ident) = b.identifier {
    panic!();//show_arrayed_identifier(f, ident);
  }
}

struct NoWrite;
impl Write for NoWrite {
  fn write_str(&mut self, s: &str) -> Result<(), std::fmt::Error> {
    Ok(())
  }
}

// This is a hack to run through the first time with an empty writter to find if 'return' is declared.
pub fn has_conditional_return(state: &mut OutputState, cst: &hir::CompoundStatement) -> bool {
  let mut f = NoWrite{};
  show_compound_statement(&mut f, state, cst);
  let result = state.return_declared;
  state.return_declared = false;
  result
}

pub fn show_function_definition<F>(f: &mut F, state: &mut OutputState, fd: &hir::FunctionDefinition) where F: Write {
  show_function_prototype(f, state, &fd.prototype);
  let _ = f.write_str(" ");
  state.return_type = Some(Box::new(fd.prototype.ty.clone()));
  state.return_declared = has_conditional_return(state, &fd.statement);

  show_indent(f, state);
  let _ = f.write_str("{\n");

  state.indent();
  if state.return_declared {
    show_indent(f, state);
    f.write_str("I32 ret_mask = ~0;\n");
    // XXX: the cloning here is bad
    show_indent(f, state);
    show_type(f, state, &state.return_type.clone().unwrap());
    f.write_str(" ret;\n");
  }
  for st in &fd.statement.statement_list {
    show_statement(f, state, st);
  }
  state.outdent();

  show_indent(f, state);
  let _ = f.write_str("}\n");
  state.return_type = None;
  state.return_declared = false;
}

pub fn show_compound_statement<F>(f: &mut F, state: &mut OutputState, cst: &hir::CompoundStatement) where F: Write {
  show_indent(f, state);
  let _ = f.write_str("{\n");

  state.indent();
  for st in &cst.statement_list {
    show_statement(f, state, st);
  }
  state.outdent();

  show_indent(f, state);
  let _ = f.write_str("}\n");
}

pub fn show_statement<F>(f: &mut F, state: &mut OutputState, st: &hir::Statement) where F: Write {
  match *st {
    hir::Statement::Compound(ref cst) => show_compound_statement(f, state, cst),
    hir::Statement::Simple(ref sst) => show_simple_statement(f, state, sst)
  }
}

pub fn show_simple_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SimpleStatement) where F: Write {
  match *sst {
    hir::SimpleStatement::Declaration(ref d) => show_declaration(f, state, d),
    hir::SimpleStatement::Expression(ref e) => show_expression_statement(f, state, e),
    hir::SimpleStatement::Selection(ref s) => show_selection_statement(f, state, s),
    hir::SimpleStatement::Switch(ref s) => show_switch_statement(f, state, s),
    hir::SimpleStatement::Iteration(ref i) => show_iteration_statement(f, state, i),
    hir::SimpleStatement::Jump(ref j) => show_jump_statement(f, state, j)
  }
}

pub fn show_indent<F>(f: &mut F, state: &mut OutputState) where F: Write {
  for i in 0..state.indent {
    let _ = f.write_str(" ");
  }
}

pub fn show_expression_statement<F>(f: &mut F, state: &mut OutputState, est: &hir::ExprStatement) where F: Write {
  show_indent(f, state);

  if let Some(ref e) = *est {
    show_hir_expr(f, state, e);
  }

  let _ = f.write_str(";\n");
}

pub fn show_selection_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SelectionStatement) where F: Write {
  if state.output_cxx {
    state.mask = Some(sst.cond.clone());
    show_selection_rest_statement(f, state, &sst.rest);
    state.mask = None;
  } else {
    show_indent(f, state);
    let _ = f.write_str("if (");
    show_hir_expr(f, state, &sst.cond);
    let _ = f.write_str(") {\n");
    state.indent();
    show_selection_rest_statement(f, state, &sst.rest);
  }
}

pub fn show_selection_rest_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SelectionRestStatement) where F: Write {
  match *sst {
    hir::SelectionRestStatement::Statement(ref if_st) => {
      show_statement(f, state, if_st);
      if !state.output_cxx {
        state.outdent();
        show_indent(f, state);
        let _ = f.write_str("}\n");
      }
    }
    hir::SelectionRestStatement::Else(ref if_st, ref else_st) => {

      show_statement(f, state, if_st);

      let previous = state.mask.clone();
      // invert the mask condition
      state.mask = state.mask.as_ref().map(|mask| Box::new(hir::Expr {
        kind: hir::ExprKind::Unary(UnaryOp::Not, mask.clone()),
        ty: hir::Type::new(hir::TypeKind::Bool) }));

      if !state.output_cxx {
        state.outdent();
        show_indent(f, state);
        let _ = f.write_str("} else ");
      }
      show_statement(f, state, else_st);
      state.mask = previous;
    }
  }
}

pub fn show_switch_statement<F>(f: &mut F, state: &mut OutputState, sst: &hir::SwitchStatement) where F: Write {
  show_indent(f, state);
  let _ = f.write_str("switch (");
  show_hir_expr(f, state, &sst.head);
  let _ = f.write_str(") {\n");
  state.indent();

  for case in &sst.cases {
    show_case_label(f, state, &case.label);
    state.indent();
    for st in &case.stmts {
      show_statement(f, state, st);
    }
    state.outdent();
  }
  state.outdent();
  show_indent(f, state);
  let _ = f.write_str("}\n");

}

pub fn show_case_label<F>(f: &mut F, state: &mut OutputState, cl: &hir::CaseLabel) where F: Write {
  show_indent(f, state);
  match *cl {
    hir::CaseLabel::Case(ref e) => {
      let _ = f.write_str("case ");
      show_hir_expr(f, state, e);
      let _ = f.write_str(":\n");
    }
    hir::CaseLabel::Def => { let _ = f.write_str("default:\n"); }
  }
}

pub fn show_iteration_statement<F>(f: &mut F, state: &mut OutputState, ist: &hir::IterationStatement) where F: Write {
  show_indent(f, state);
  match *ist {
    hir::IterationStatement::While(ref cond, ref body) => {
      let _ = f.write_str("while (");
      show_condition(f, state, cond);
      let _ = f.write_str(") ");
      show_statement(f, state, body);
    }
    hir::IterationStatement::DoWhile(ref body, ref cond) => {
      let _ = f.write_str("do ");
      show_statement(f, state, body);
      let _ = f.write_str(" while (");
      show_hir_expr(f, state, cond);
      let _ = f.write_str(")\n");
    }
    hir::IterationStatement::For(ref init, ref rest, ref body) => {
      let _ = f.write_str("for (");
      show_for_init_statement(f, state, init);
      show_for_rest_statement(f, state, rest);
      let _ = f.write_str(") ");
      show_statement(f, state, body);
    }
  }
}

pub fn show_condition<F>(f: &mut F, state: &mut OutputState, c: &hir::Condition) where F: Write {
  match *c {
    hir::Condition::Expr(ref e) => show_hir_expr(f, state, e),
    /*hir::Condition::Assignment(ref ty, ref name, ref initializer) => {
      show_type(f, state, ty);
      let _ = f.write_str(" ");
      show_identifier(f, name);
      let _ = f.write_str(" = ");
      show_initializer(f, state, initializer);
    }*/
  }
}

pub fn show_for_init_statement<F>(f: &mut F, state: &mut OutputState, i: &hir::ForInitStatement) where F: Write {
  match *i {
    hir::ForInitStatement::Expression(ref expr) => {
      if let Some(ref e) = *expr {
        show_hir_expr(f, state, e);
      }
    }
    hir::ForInitStatement::Declaration(ref d) => {
      state.in_loop_declaration = true;
      show_declaration(f, state, d);
      state.in_loop_declaration = false;

    }
  }
}

pub fn show_for_rest_statement<F>(f: &mut F, state: &mut OutputState, r: &hir::ForRestStatement) where F: Write {
  if let Some(ref cond) = r.condition {
    show_condition(f, state, cond);
  }

  let _ = f.write_str("; ");

  if let Some(ref e) = r.post_expr {
    show_hir_expr(f, state, e);
  }
}

pub fn show_jump_statement<F>(f: &mut F, state: &mut OutputState, j: &hir::JumpStatement) where F: Write {
  show_indent(f, state);
  match *j {
    hir::JumpStatement::Continue => { let _ = f.write_str("continue;\n"); }
    hir::JumpStatement::Break => { let _ = f.write_str("break;\n"); }
    hir::JumpStatement::Discard => { let _ = f.write_str("discard;\n"); }
    hir::JumpStatement::Return(ref e) => {
      if state.output_cxx {
        if state.mask.is_some() {
          if !state.return_declared {
            // XXX: if we're nested then this declaration won't work
            f.write_str("I32 ret_mask = ~0;\n");
            // XXX: the cloning here is bad
            show_type(f, state, &state.return_type.clone().unwrap());
            f.write_str(" ret;\n");
            state.return_declared = true;
          }
          // XXX: the cloning here is bad
          let _ = f.write_str("ret = if_then_else(ret_mask & (");
          show_hir_expr(f, state, &state.mask.clone().unwrap());
          let _ = f.write_str("), ");
          show_hir_expr(f, state, e);
          let _ = f.write_str(", ret);\n");
          show_indent(f, state);
          let _ = f.write_str("ret_mask &= ~(");
          show_hir_expr(f, state, &state.mask.clone().unwrap());
          let _ = f.write_str(");\n");
        } else {
          if state.return_declared {
            let _  = f.write_str("return if_then_else(ret_mask, ");
            show_hir_expr(f, state, e);
            let _  = f.write_str(", ret);\n");
          } else {
            let _ = f.write_str("return ");
            show_hir_expr(f, state, e);
            let _ = f.write_str(";\n");
          }
        }
      } else {
        let _ = f.write_str("return ");
        show_hir_expr(f, state, e);
        let _ = f.write_str(";\n");
      }
    }
  }
}

pub fn show_preprocessor<F>(f: &mut F, pp: &hir::Preprocessor) where F: Write {
  match *pp {
    hir::Preprocessor::Define(ref pd) => show_preprocessor_define(f, pd),
    hir::Preprocessor::Version(ref pv) => show_preprocessor_version(f, pv),
    hir::Preprocessor::Extension(ref pe) => show_preprocessor_extension(f, pe)
  }
}

pub fn show_preprocessor_define<F>(f: &mut F, pd: &hir::PreprocessorDefine) where F: Write {
  let _ = write!(f, "#define {} ", pd.name);
  show_expr(f, panic!());
  let _ = f.write_str("\n");
}

pub fn show_preprocessor_version<F>(f: &mut F, pv: &hir::PreprocessorVersion) where F: Write {
  let _ = write!(f, "#version {}", pv.version);

  if let Some(ref profile) = pv.profile {
    match *profile {
      hir::PreprocessorVersionProfile::Core => { let _ = f.write_str(" core"); }
      hir::PreprocessorVersionProfile::Compatibility => { let _ = f.write_str(" compatibility"); }
      hir::PreprocessorVersionProfile::ES => { let _ = f.write_str(" es"); }
    }
  }

  let _ = f.write_str("\n");
}

pub fn show_preprocessor_extension<F>(f: &mut F, pe: &hir::PreprocessorExtension) where F: Write {
  let _ = f.write_str("#extension ");

  match pe.name {
    hir::PreprocessorExtensionName::All => { let _ = f.write_str("all"); }
    hir::PreprocessorExtensionName::Specific(ref n) => { let _ = f.write_str(n); }
  }

  if let Some(ref behavior) = pe.behavior {
    match *behavior {
      hir::PreprocessorExtensionBehavior::Require => { let _ = f.write_str(" : require"); }
      hir::PreprocessorExtensionBehavior::Enable => { let _ = f.write_str(" : enable"); }
      hir::PreprocessorExtensionBehavior::Warn => { let _ = f.write_str(" : warn"); }
      hir::PreprocessorExtensionBehavior::Disable => { let _ = f.write_str(" : disable"); }
    }
  }

  let _ = f.write_str("\n");
}

pub fn show_external_declaration<F>(f: &mut F, state: &mut OutputState, ed: &hir::ExternalDeclaration) where F: Write {
  match *ed {
    hir::ExternalDeclaration::Preprocessor(ref pp) => show_preprocessor(f, pp),
  hir::ExternalDeclaration::FunctionDefinition(ref fd) => show_function_definition(f, state, fd),
  hir::ExternalDeclaration::Declaration(ref d) => show_declaration(f, state, d)
  }
}

pub fn show_translation_unit<F>(f: &mut F, state: &mut OutputState, tu: &hir::TranslationUnit) where F: Write {
  for ed in &(tu.0).0 {
    show_external_declaration(f, state, ed);
  }
}

