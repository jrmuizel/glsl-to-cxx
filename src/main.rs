use glsl_to_cxx::translate;
fn main() {
    translate(&std::env::args());
}
