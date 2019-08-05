use std::fmt;
use std::iter::{FromIterator, once};
use std::ops::{Deref, DerefMut};
use std::collections::HashMap;

use glsl::syntax::{NonEmpty, TypeQualifier, TypeSpecifier, TypeSpecifierNonArray};
use glsl::syntax;
use glsl::syntax::StructFieldSpecifier;
use glsl::syntax::PrecisionQualifier;
use glsl::syntax::ArrayedIdentifier;
use glsl::syntax::FullySpecifiedType;
use glsl::syntax::ArraySpecifier;
use glsl::syntax::TypeName;
use glsl::syntax::StructSpecifier;
use glsl::syntax::UnaryOp;
use glsl::syntax::BinaryOp;
use glsl::syntax::AssignmentOp;
use glsl::syntax::Identifier;
use crate::hir::Type::Function;
use crate::hir::Initializer::Simple;
use crate::hir::SimpleStatement::Jump;

#[derive(Debug)]
pub struct Symbol {
    pub name: String,
    ty: Type
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    ret: Box<Type>,
    params: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    signatures: NonEmpty<FunctionSignature>
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Function(FunctionType),
    FullySpecifiedType(FullySpecifiedType)
}

impl Type {
    fn new(t: TypeSpecifierNonArray) -> Self {
        Type::FullySpecifiedType(FullySpecifiedType::new(t))
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct SymRef(u32);

#[derive(Debug)]
struct Scope {
    name: String,
    names: HashMap<String, SymRef>,
}
impl Scope {
    fn new(name: String) -> Self {  Scope { name, names: HashMap::new() }}
}

#[derive(Debug)]
pub struct State {
    scopes: Vec<Scope>,
    syms: Vec<Symbol>,
}


impl State {
    pub fn new() -> Self {
        State { scopes: Vec::new(), syms: Vec::new() }
    }

    fn lookup(&self, name: &str) -> Option<SymRef> {
        for s in self.scopes.iter().rev() {
            if let Some(sym) = s.names.get(name) {
                return Some(*sym);
            }
        }
        return None;
    }

    fn declare(&mut self, name: &str, ty: Type) {
        let s = SymRef(self.syms.len() as u32);
        self.syms.push(Symbol{ name: name.into(), ty});
        self.scopes.last_mut().unwrap().names.insert(name.into(), s);
    }

    pub fn sym(&self, sym: SymRef) -> &Symbol {
        &self.syms[sym.0 as usize]
    }

    pub fn lookup_sym_mut(&mut self, name: &str) -> Option<&mut Symbol> {
        self.lookup(name).map(move |x| &mut self.syms[x.0 as usize])
    }

    fn push_scope(&mut self, name: String) {
        self.scopes.push(Scope::new(name));
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}




/// Error that might occur when creating a new [`Identifier`].
#[derive(Debug)]
pub enum IdentifierError {
    StartsWithDigit,
    ContainsNonASCIIAlphaNum
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            IdentifierError::StartsWithDigit =>
                f.write_str("starts starts with a digit"),

            IdentifierError::ContainsNonASCIIAlphaNum =>
                f.write_str("contains at least one non-alphanumeric ASCII character")
        }
    }
}




/// A declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum Declaration {
    FunctionPrototype(FunctionPrototype),
    InitDeclaratorList(InitDeclaratorList),
    Precision(PrecisionQualifier, TypeSpecifier),
    Block(Block),
    Global(TypeQualifier, Vec<Identifier>)
}

/// A general purpose block, containing fields and possibly a list of declared identifiers. Semantic
/// is given with the storage qualifier.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub qualifier: TypeQualifier,
    pub name: Identifier,
    pub fields: Vec<StructFieldSpecifier>,
    pub identifier: Option<ArrayedIdentifier>
}

/// Function identifier.
#[derive(Clone, Debug, PartialEq)]
pub enum FunIdentifier {
    Identifier(SymRef),
    Expr(Box<Expr>)
}

/// Function prototype.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionPrototype {
    pub ty: FullySpecifiedType,
    pub name: Identifier,
    pub parameters: Vec<FunctionParameterDeclaration>
}

/// Function parameter declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionParameterDeclaration {
    Named(Option<TypeQualifier>, FunctionParameterDeclarator),
    Unnamed(Option<TypeQualifier>, TypeSpecifier)
}

impl FunctionParameterDeclaration {
    /// Create a named function argument.
    pub fn new_named<I, T>(
        ident: I,
        ty: T
    ) -> Self
        where I: Into<ArrayedIdentifier>,
              T: Into<TypeSpecifier> {
        let declator = FunctionParameterDeclarator {
            ty: ty.into(),
            ident: ident.into()
        };

        FunctionParameterDeclaration::Named(None, declator)
    }

    /// Create an unnamed function argument (mostly useful for interfaces / function prototypes).
    pub fn new_unnamed<T>(ty: T) -> Self where T: Into<TypeSpecifier> {
        FunctionParameterDeclaration::Unnamed(None, ty.into())
    }
}

/// Function parameter declarator.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionParameterDeclarator {
    pub ty: TypeSpecifier,
    pub ident: ArrayedIdentifier
}

/// Init declarator list.
#[derive(Clone, Debug, PartialEq)]
pub struct InitDeclaratorList {
    pub head: SingleDeclaration,
    pub tail: Vec<SingleDeclarationNoType>
}

/// Single declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleDeclaration {
    pub ty: FullySpecifiedType,
    pub name: Option<Identifier>,
    pub array_specifier: Option<ArraySpecifier>,
    pub initializer: Option<Initializer>
}

/// A single declaration with implicit, already-defined type.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleDeclarationNoType {
    pub ident: ArrayedIdentifier,
    pub initializer: Option<Initializer>
}

/// Initializer.
#[derive(Clone, Debug, PartialEq)]
pub enum Initializer {
    Simple(Box<Expr>),
    List(NonEmpty<Initializer>)
}

impl From<Expr> for Initializer {
    fn from(e: Expr) -> Self {
        Initializer::Simple(Box::new(e))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type
}


/// The most general form of an expression. As you can see if you read the variant list, in GLSL, an
/// assignment is an expression. This is a bit silly but think of an assignment as a statement first
/// then an expression which evaluates to what the statement “returns”.
///
/// An expression is either an assignment or a list (comma) of assignments.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// A variable expression, using an identifier.
    Variable(SymRef),
    /// Integral constant expression.
    IntConst(i32),
    /// Unsigned integral constant expression.
    UIntConst(u32),
    /// Boolean constant expression.
    BoolConst(bool),
    /// Single precision floating expression.
    FloatConst(f32),
    /// Double precision floating expression.
    DoubleConst(f64),
    /// A unary expression, gathering a single expression and a unary operator.
    Unary(UnaryOp, Box<Expr>),
    /// A binary expression, gathering two expressions and a binary operator.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// A ternary conditional expression, gathering three expressions.
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    /// An assignment is also an expression. Gathers an expression that defines what to assign to, an
    /// assignment operator and the value to associate with.
    Assignment(Box<Expr>, AssignmentOp, Box<Expr>),
    /// Add an array specifier to an expression.
    Bracket(Box<Expr>, ArraySpecifier),
    /// A functional call. It has a function identifier and a list of expressions (arguments).
    FunCall(FunIdentifier, Vec<Expr>),
    /// An expression associated with a field selection (struct).
    Dot(Box<Expr>, Identifier),
    /// Post-incrementation of an expression.
    PostInc(Box<Expr>),
    /// Post-decrementation of an expression.
    PostDec(Box<Expr>),
    /// An expression that contains several, separated with comma.
    Comma(Box<Expr>, Box<Expr>)
}

/*
impl From<i32> for Expr {
    fn from(x: i32) -> Expr {
        ExprKind::IntConst(x)
    }
}

impl From<u32> for Expr {
    fn from(x: u32) -> Expr {
        Expr::UIntConst(x)
    }
}

impl From<bool> for Expr {
    fn from(x: bool) -> Expr {
        Expr::BoolConst(x)
    }
}

impl From<f32> for Expr {
    fn from(x: f32) -> Expr {
        Expr::FloatConst(x)
    }
}

impl From<f64> for Expr {
    fn from(x: f64) -> Expr {
        Expr::DoubleConst(x)
    }
}
*/
/// Starting rule.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationUnit(pub NonEmpty<ExternalDeclaration>);

impl TranslationUnit {
    /// Construct a translation unit from an iterator.
    ///
    /// # Errors
    ///
    /// `None` if the iterator yields no value.
    pub fn from_iter<I>(iter: I) -> Option<Self> where I: IntoIterator<Item = ExternalDeclaration> {
        NonEmpty::from_iter(iter).map(TranslationUnit)
    }
}

impl Deref for TranslationUnit {
    type Target = NonEmpty<ExternalDeclaration>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TranslationUnit {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for TranslationUnit {
    type IntoIter = <NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a TranslationUnit {
    type IntoIter = <&'a NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = &'a ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter()
    }
}

impl<'a> IntoIterator for &'a mut TranslationUnit {
    type IntoIter = <&'a mut NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = &'a mut ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.0).into_iter()
    }
}

/// External declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum ExternalDeclaration {
    Preprocessor(Preprocessor),
    FunctionDefinition(FunctionDefinition),
    Declaration(Declaration)
}

impl ExternalDeclaration {
    /// Create a new function.
    pub fn new_fn<T, N, A, S>(
        ret_ty: T,
        name: N,
        args: A,
        body: S
    ) -> Self
        where T: Into<FullySpecifiedType>,
              N: Into<Identifier>,
              A: IntoIterator<Item = FunctionParameterDeclaration>,
              S: IntoIterator<Item = Statement> {
        ExternalDeclaration::FunctionDefinition(
            FunctionDefinition {
                prototype: FunctionPrototype {
                    ty: ret_ty.into(),
                    name: name.into(),
                    parameters: args.into_iter().collect()
                },
                statement: CompoundStatement {
                    statement_list: body.into_iter().collect()
                }
            }
        )
    }

    /// Create a new structure.
    ///
    /// # Errors
    ///
    ///   - [`None`] if no fields are provided. GLSL forbids having empty structs.
    pub fn new_struct<N, F>(name: N, fields: F) -> Option<Self>
        where N: Into<TypeName>,
              F: IntoIterator<Item = StructFieldSpecifier> {
        let fields: Vec<_> = fields.into_iter().collect();

        if fields.is_empty() {
            None
        } else {
            Some(ExternalDeclaration::Declaration(
                Declaration::InitDeclaratorList(
                    InitDeclaratorList {
                        head: SingleDeclaration {
                            ty: FullySpecifiedType {
                                qualifier: None,
                                ty: TypeSpecifier {
                                    ty: TypeSpecifierNonArray::Struct(
                                        StructSpecifier {
                                            name: Some(name.into()),
                                            fields: NonEmpty(fields.into_iter().collect())
                                        }
                                    ),
                                    array_specifier: None
                                }
                            },
                            name: None,
                            array_specifier: None,
                            initializer: None
                        },
                        tail: vec![]
                    }
                )
            ))
        }
    }
}

/// Function definition.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDefinition {
    pub prototype: FunctionPrototype,
    pub statement: CompoundStatement,
}

/// Compound statement (with no new scope).
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundStatement {
    pub statement_list: Vec<Statement>
}

impl FromIterator<Statement> for CompoundStatement {
    fn from_iter<T>(iter: T) -> Self where T: IntoIterator<Item = Statement> {
        CompoundStatement {
            statement_list: iter.into_iter().collect()
        }
    }
}

/// Statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Compound(Box<CompoundStatement>),
    Simple(Box<SimpleStatement>)
}

impl Statement {
    /// Create a case-label sequence of nested statements.
    pub fn new_case<C, S>(
        case: C,
        statements: S
    ) -> Self
        where C: Into<CaseLabel>,
              S: IntoIterator<Item = Statement> {
        let case_stmt = Statement::Simple(Box::new(SimpleStatement::CaseLabel(case.into())));

        Statement::Compound(
            Box::new(CompoundStatement {
                statement_list: once(case_stmt).chain(statements.into_iter()).collect()
            })
        )
    }

    /// Declare a new variable.
    ///
    /// `ty` is the type of the variable, `name` the name of the binding to create,
    /// `array_specifier` an optional argument to make your binding an array and
    /// `initializer`
    pub fn declare_var<T, N, A, I>(
        ty: T,
        name: N,
        array_specifier: A,
        initializer: I
    ) -> Self
        where T: Into<FullySpecifiedType>,
              N: Into<Identifier>,
              A: Into<Option<ArraySpecifier>>,
              I: Into<Option<Initializer>> {
        Statement::Simple(
            Box::new(SimpleStatement::Declaration(
                Declaration::InitDeclaratorList(
                    InitDeclaratorList {
                        head: SingleDeclaration {
                            ty: ty.into(),
                            name: Some(name.into()),
                            array_specifier: array_specifier.into(),
                            initializer: initializer.into()
                        },
                        tail: Vec::new()
                    }
                )
            ))
        )
    }
}

/// Simple statement.
#[derive(Clone, Debug, PartialEq)]
pub enum SimpleStatement {
    Declaration(Declaration),
    Expression(ExprStatement),
    Selection(SelectionStatement),
    Switch(SwitchStatement),
    CaseLabel(CaseLabel),
    Iteration(IterationStatement),
    Jump(JumpStatement)
}

impl SimpleStatement {
    /// Create a new expression statement.
    pub fn new_expr<E>(expr: E) -> Self where E: Into<Expr> {
        SimpleStatement::Expression(Some(expr.into()))
    }

    /// Create a new selection statement (if / else).
    pub fn new_if_else<If, True, False>(
        ife: If,
        truee: True,
        falsee: False
    ) -> Self
        where If: Into<Expr>,
              True: Into<Statement>,
              False: Into<Statement> {
        SimpleStatement::Selection(
            SelectionStatement {
                cond: Box::new(ife.into()),
                rest: SelectionRestStatement::Else(Box::new(truee.into()), Box::new(falsee.into()))
            }
        )
    }

    /// Create a new switch statement.
    ///
    /// A switch statement is always composed of a [`SimpleStatement::Switch`] block, that contains it
    /// all, and has as body a compound list of case statements.
    pub fn new_switch<H, B>(
        head: H,
        body: B
    ) -> Self
        where H: Into<Expr>,
              B: IntoIterator<Item = Statement> {
        SimpleStatement::Switch(
            SwitchStatement {
                head: Box::new(head.into()),
                body: body.into_iter().collect()
            }
        )
    }

    /// Create a new while statement.
    pub fn new_while<C, S>(
        cond: C,
        body: S
    ) -> Self
        where C: Into<Condition>,
              S: Into<Statement> {
        SimpleStatement::Iteration(
            IterationStatement::While(cond.into(), Box::new(body.into()))
        )
    }

    /// Create a new do-while statement.
    pub fn new_do_while<C, S>(
        body: S,
        cond: C
    ) -> Self
        where S: Into<Statement>,
              C: Into<Expr> {
        SimpleStatement::Iteration(
            IterationStatement::DoWhile(Box::new(body.into()), Box::new(cond.into()))
        )
    }
}

/// Expression statement.
pub type ExprStatement = Option<Expr>;

/// Selection statement.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionStatement {
    pub cond: Box<Expr>,
    pub rest: SelectionRestStatement
}

/// Condition.
#[derive(Clone, Debug, PartialEq)]
pub enum Condition {
    Expr(Box<Expr>),
    Assignment(FullySpecifiedType, Identifier, Initializer)
}

impl From<Expr> for Condition {
    fn from(expr: Expr) -> Self {
        Condition::Expr(Box::new(expr))
    }
}

/// Selection rest statement.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectionRestStatement {
    /// Body of the if.
    Statement(Box<Statement>),
    /// The first argument is the body of the if, the rest is the next statement.
    Else(Box<Statement>, Box<Statement>)
}

/// Switch statement.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchStatement {
    pub head: Box<Expr>,
    pub body: Vec<Statement>
}

/// Case label statement.
#[derive(Clone, Debug, PartialEq)]
pub enum CaseLabel {
    Case(Box<Expr>),
    Def
}

/// Iteration statement.
#[derive(Clone, Debug, PartialEq)]
pub enum IterationStatement {
    While(Condition, Box<Statement>),
    DoWhile(Box<Statement>, Box<Expr>),
    For(ForInitStatement, ForRestStatement, Box<Statement>)
}

/// For init statement.
#[derive(Clone, Debug, PartialEq)]
pub enum ForInitStatement {
    Expression(Option<Expr>),
    Declaration(Box<Declaration>)
}

/// For init statement.
#[derive(Clone, Debug, PartialEq)]
pub struct ForRestStatement {
    pub condition: Option<Condition>,
    pub post_expr: Option<Box<Expr>>
}

/// Jump statement.
#[derive(Clone, Debug, PartialEq)]
pub enum JumpStatement {
    Continue,
    Break,
    Return(Box<Expr>),
    Discard
}

/// Some basic preprocessor commands.
///
/// As it’s important to carry them around the AST because they cannot be substituted in a normal
/// preprocessor (they’re used by GPU’s compilers), those preprocessor commands are available for
/// inspection.
///
/// > Important note: so far, only `#version` and `#extension` are supported. Other pragmas will be
/// > added in the future. Stay tuned.
#[derive(Clone, Debug, PartialEq)]
pub enum Preprocessor {
    Define(PreprocessorDefine),
    Version(PreprocessorVersion),
    Extension(PreprocessorExtension)
}

/// A #define preprocessor command.
/// Allows any expression but only Integer and Float literals make sense
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorDefine {
    pub name: Identifier,
    pub value: Expr,
}

/// A #version preprocessor command.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorVersion {
    pub version: u16,
    pub profile: Option<PreprocessorVersionProfile>
}

/// A #version profile annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorVersionProfile {
    Core,
    Compatibility,
    ES
}

/// An #extension preprocessor command.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorExtension {
    pub name: PreprocessorExtensionName,
    pub behavior: Option<PreprocessorExtensionBehavior>
}

/// An #extension name annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorExtensionName {
    /// All extensions you could ever imagine in your whole lifetime (how crazy is that!).
    All,
    /// A specific extension.
    Specific(String)
}

/// An #extension behavior annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorExtensionBehavior {
    Require,
    Enable,
    Warn,
    Disable
}

trait NonEmptyExt<T> {
    fn map<U, F: FnMut(&mut State, &T) -> U>(&self, s: &mut State, f: F) -> NonEmpty<U>;
    fn new(x: T) -> NonEmpty<T>;
}

impl<T> NonEmptyExt<T> for NonEmpty<T> {
    fn map<U, F: FnMut(&mut State, &T) -> U>(&self, s: &mut State, mut f: F) -> NonEmpty<U> {
        NonEmpty::from_iter(self.into_iter().map(|x| f(s, &x))).unwrap()
    }
    fn new(x: T) -> NonEmpty<T> {
        NonEmpty::from_iter(vec![x].into_iter()).unwrap()
    }
}


fn translate_initializater(state: &mut State, i: &syntax::Initializer) -> Initializer {
    match i {
        syntax::Initializer::Simple(i) => Initializer::Simple(Box::new(translate_expression(state, i))),
        _ => panic!()
    }
}

fn translate_single_declaration(state: &mut State, d: &syntax::SingleDeclaration) -> SingleDeclaration {
    let mut ty = d.ty.clone();
    ty.ty.array_specifier = d.array_specifier.clone();
    state.declare(d.name.as_ref().unwrap().as_str(), Type::FullySpecifiedType(ty.clone()));
    SingleDeclaration {
        name: d.name.clone(),
        ty,
        array_specifier: d.array_specifier.clone(),
        initializer: d.initializer.as_ref().map(|x| translate_initializater(state, x)),
    }
}

fn translate_single_declaration_no_type(state: &mut State, d: &syntax::SingleDeclarationNoType) -> SingleDeclarationNoType {
    panic!()
}

fn translate_init_declarator_list(state: &mut State, l: &syntax::InitDeclaratorList) -> InitDeclaratorList {
    InitDeclaratorList {
        head: translate_single_declaration(state, &l.head),
        tail: l.tail.iter().map(|x| translate_single_declaration_no_type(state, x)).collect()
    }
}

fn translate_declaration(state: &mut State, d: &syntax::Declaration) -> Declaration {
    match d {
        syntax::Declaration::Block(b) => Declaration::Block(panic!()),
        syntax::Declaration::FunctionPrototype(p) => Declaration::FunctionPrototype(panic!()),
        syntax::Declaration::Global(ty, ids) => Declaration::Global(panic!(), panic!()),
        syntax::Declaration::InitDeclaratorList(dl) => Declaration::InitDeclaratorList(translate_init_declarator_list(state, dl)),
        syntax::Declaration::Precision(p, ts) => Declaration::Precision(panic!(), panic!()),
    }
}

fn is_vector(ty: &Type) -> bool {
    match ty {
        Type::FullySpecifiedType(FullySpecifiedType { ty, .. }) => {
            match ty.ty {
                TypeSpecifierNonArray::Vec3 | TypeSpecifierNonArray::Vec2 | TypeSpecifierNonArray::Vec4 => {
                    true
                }
                _ => false
            }
        }
        _ => false
    }
}

fn compatible_type(lhs: &Type, rhs: &Type) -> bool {
    if lhs == &Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Double)) &&
        rhs == &Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Float)) {
        true
    } else if rhs == &Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Double)) &&
        lhs == &Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Float)) {
        true
    } else {
        match (lhs, rhs) {
            (Type::FullySpecifiedType(lhs), Type::FullySpecifiedType(rhs)) => {
                lhs.ty == rhs.ty
            }
            _ => panic!("unexpected type")
        }
    }
}

fn promoted_type(lhs: &Type, rhs: &Type) -> Type {
    if lhs == &Type::new(TypeSpecifierNonArray::Double) &&
        rhs == &Type::new(TypeSpecifierNonArray::Float) {
        Type::new(TypeSpecifierNonArray::Double)
    } else if lhs == &Type::new(TypeSpecifierNonArray::Float) &&
        rhs == &Type::new(TypeSpecifierNonArray::Double) {
        Type::new(TypeSpecifierNonArray::Double)
    } else if is_vector(&lhs) && (rhs == &Type::new(TypeSpecifierNonArray::Float) ||
        rhs == &Type::new(TypeSpecifierNonArray::Double)) {
        // scalars promote to vectors
        lhs.clone()
    } else if is_vector(&rhs) && (lhs == &Type::new(TypeSpecifierNonArray::Float) ||
        lhs == &Type::new(TypeSpecifierNonArray::Double)) {
        // scalars promote to vectors
        rhs.clone()
    } else {
        match (lhs, rhs) {
            (Type::FullySpecifiedType(lhs), Type::FullySpecifiedType(rhs)) => {
                assert_eq!(lhs.ty, rhs.ty)
            }
            _ => panic!("unexpected types {:?} {:?}", lhs, rhs)
        }
        lhs.clone()
    }
}

fn translate_expression(state: &mut State, e: &syntax::Expr) -> Expr {
    match e {
        syntax::Expr::Variable(i) => {
            let sym = match state.lookup(i.as_str()) {
                Some(sym) => sym,
                None => panic!("missing declaration {}", i.as_str())
            };
            Expr { kind: ExprKind::Variable(sym), ty: state.sym(sym).ty.clone() }
        },
        syntax::Expr::Assignment(lhs, op, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert!(compatible_type(&lhs.ty, &rhs.ty));
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Assignment(lhs, op.clone(), rhs), ty }
        }
        syntax::Expr::Binary(op, lhs, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            let ty = if op == &BinaryOp::Mult {
                match (&lhs.ty, &rhs.ty) {
                    (Type::FullySpecifiedType(FullySpecifiedType { ty: ilhs, ..}),
                     Type::FullySpecifiedType(FullySpecifiedType { ty: irhs, ..})) => {
                        if ilhs.ty == TypeSpecifierNonArray::Mat3 && irhs.ty == TypeSpecifierNonArray::Vec3 {
                            rhs.ty.clone()
                        } else {
                            promoted_type(&lhs.ty, &rhs.ty)
                        }
                    }
                    _ => promoted_type(&lhs.ty, &rhs.ty)
                }
            } else {
                promoted_type(&lhs.ty, &rhs.ty)
            };
            Expr { kind: ExprKind::Binary(op.clone(), lhs, rhs), ty}
        }
        syntax::Expr::Unary(op, e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::Unary(op.clone(), e), ty}
        }
        syntax::Expr::BoolConst(b) => {
            Expr { kind: ExprKind::BoolConst(*b), ty: Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Bool)) }
        }
        syntax::Expr::Comma(lhs, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert_eq!(lhs.ty, rhs.ty);
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Comma(lhs, rhs), ty }
        }
        syntax::Expr::DoubleConst(d) => {
            Expr { kind: ExprKind::DoubleConst(*d), ty: Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Double)) }
        }
        syntax::Expr::FloatConst(f) => {
            Expr { kind: ExprKind::FloatConst(*f), ty: Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Float)) }
        },
        syntax::Expr::FunCall(fun, params) => {
            let ty: Type;
            let params: Vec<Expr> = params.iter().map(|x| translate_expression(state, x)).collect();
            Expr {
                kind:
                ExprKind::FunCall(
                    match fun {
                        syntax::FunIdentifier::Identifier(i) => {
                            let sym = match state.lookup(i.as_str()) {
                                Some(s) => s,
                                None => panic!("missing {}", i.as_str())
                            };
                            let fn_ty = match &state.sym(sym).ty {
                                Type::Function(f) => f,
                                _ => panic!("can only call functions")
                            };
                            let mut ret = None;
                            for sig in &fn_ty.signatures {
                                let mut matching = true;
                                for (e, p) in params.iter().zip(sig.params.iter()) {
                                    if !compatible_type(&e.ty, p) {
                                        matching = false;
                                        break;
                                    }
                                }
                                if matching {
                                    ret = Some(sig.ret.clone());
                                    break;
                                }
                            }
                            ty = match ret {
                                Some(t) => *t,
                                None => {
                                    dbg!(&fn_ty.signatures);
                                    dbg!(params.iter().map(|p| &p.ty).collect::<Vec<_>>());
                                    panic!("no matching func {}", i.as_str())
                                }
                            };
                            FunIdentifier::Identifier(sym)
                        },
                        _ => panic!()
                    },
                    params
                ),
                ty,
            }
        }
        syntax::Expr::IntConst(i) => {
            Expr { kind: ExprKind::IntConst(*i), ty: Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::Int)) }
        }
        syntax::Expr::UIntConst(u) => {
            Expr { kind: ExprKind::UIntConst(*u), ty: Type::FullySpecifiedType(FullySpecifiedType::new(TypeSpecifierNonArray::UInt)) }
        }
        syntax::Expr::PostDec(e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::PostDec(e), ty }
        }
        syntax::Expr::PostInc(e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::PostInc(e), ty }
        }
        syntax::Expr::Ternary(cond, lhs, rhs) => {
            let cond = Box::new(translate_expression(state, cond));
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert_eq!(lhs.ty, rhs.ty);
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Ternary(cond, lhs, rhs), ty }
        }
        syntax::Expr::Dot(e, i) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            if is_vector(&ty) {
                let ty = Type::FullySpecifiedType(FullySpecifiedType::new(match i.as_str().len() {
                    1 => TypeSpecifierNonArray::Float,
                    2 => TypeSpecifierNonArray::Vec2,
                    3 => TypeSpecifierNonArray::Vec3,
                    4 => TypeSpecifierNonArray::Vec4,
                    _ => panic!(),
                }));

                Expr { kind: ExprKind::Dot(e, i.clone()), ty }
            } else {
                panic!();
                Expr { kind: ExprKind::Dot(e, i.clone()), ty }
            }
        }
        syntax::Expr::Bracket(e, specifier) =>{
            let e = Box::new(translate_expression(state, e));
            let ty = if is_vector(&e.ty) {
                Type::new(TypeSpecifierNonArray::Float)
            } else {
                match &e.ty {
                    Type::FullySpecifiedType(f) => {
                        assert!(f.ty.array_specifier.is_some());
                        e.ty.clone()
                    }
                    _ => panic!("non vector array {:?} {:?}", state.sym(SymRef(17)), e)

                }
            };
            Expr { kind: ExprKind::Bracket(e, specifier.clone()), ty }
        }
    }
}

fn translate_switch(state: &mut State, s: &syntax::SwitchStatement) -> SwitchStatement {
    SwitchStatement {
        head: Box::new(translate_expression(state, &s.head)),
        body: s.body.iter().map(|x| translate_statement(state, x)).collect()
    }
}

fn translate_jump(state: &mut State, s: &syntax::JumpStatement) -> JumpStatement {
    match s {
        syntax::JumpStatement::Break => JumpStatement::Break,
        syntax::JumpStatement::Continue => JumpStatement::Continue,
        syntax::JumpStatement::Discard => JumpStatement::Discard,
        syntax::JumpStatement::Return(e) => JumpStatement::Return(Box::new(translate_expression(state, e)))
    }
}

fn translate_condition(state: &mut State, c: &syntax::Condition) -> Condition {
    match c {
        syntax::Condition::Expr(e) => Condition::Expr(Box::new(translate_expression(state, e))),
        _ => panic!()
    }
}

fn translate_for_init(state: &mut State, s: &syntax::ForInitStatement) -> ForInitStatement {
    match s {
        syntax::ForInitStatement::Expression(e) => ForInitStatement::Expression(e.as_ref().map(|e| translate_expression(state, e))),
        syntax::ForInitStatement::Declaration(d) => ForInitStatement::Declaration(Box::new(translate_declaration(state, d))),
    }
}

fn translate_for_rest(state: &mut State, s: &syntax::ForRestStatement) -> ForRestStatement {
    ForRestStatement {
        condition: s.condition.as_ref().map(|c| translate_condition(state, c)),
        post_expr: s.post_expr.as_ref().map(|e| Box::new(translate_expression(state, e)))
    }
}

fn translate_iteration(state: &mut State, s: &syntax::IterationStatement) -> IterationStatement {
    match s {
        syntax::IterationStatement::While(cond, s) =>
            IterationStatement::While(translate_condition(state, cond), Box::new(translate_statement(state, s))),
        syntax::IterationStatement::For(init, rest, s) =>
            IterationStatement::For(translate_for_init(state, init),translate_for_rest(state, rest), Box::new(translate_statement(state, s))),
        syntax::IterationStatement::DoWhile(s, e) =>
            IterationStatement::DoWhile(Box::new(translate_statement(state, s)), Box::new(translate_expression(state, e))),
    }
}

fn translate_case(state: &mut State, c: &syntax::CaseLabel) -> CaseLabel {
    match c {
        syntax::CaseLabel::Def => CaseLabel::Def,
        syntax::CaseLabel::Case(e) => CaseLabel::Case(Box::new(translate_expression(state, e)))
    }
}

fn translate_selection_rest(state: &mut State, s: &syntax::SelectionRestStatement) -> SelectionRestStatement {
    match s {
        syntax::SelectionRestStatement::Statement(s) => SelectionRestStatement::Statement(Box::new(translate_statement(state, s))),
        syntax::SelectionRestStatement::Else(if_body, rest) => {
            SelectionRestStatement::Else(Box::new(translate_statement(state, if_body)), Box::new(translate_statement(state, rest)))
        }
    }
}

fn translate_selection(state: &mut State, s: &syntax::SelectionStatement) -> SelectionStatement {
    SelectionStatement {
        cond: Box::new(translate_expression(state, &s.cond)),
        rest: translate_selection_rest(state, &s.rest),
    }
}

fn translate_simple_statement(state: &mut State, s: &syntax::SimpleStatement) -> SimpleStatement {
    match s {
        syntax::SimpleStatement::Declaration(d) => SimpleStatement::Declaration(translate_declaration(state, d)),
        syntax::SimpleStatement::CaseLabel(c) => SimpleStatement::CaseLabel(translate_case(state, c)),
        syntax::SimpleStatement::Expression(e) => SimpleStatement::Expression(e.as_ref().map(|e| translate_expression(state, e))),
        syntax::SimpleStatement::Iteration(i) => SimpleStatement::Iteration(translate_iteration(state, i)),
        syntax::SimpleStatement::Selection(s) => SimpleStatement::Selection(translate_selection(state, s)),
        syntax::SimpleStatement::Jump(j) => SimpleStatement::Jump(translate_jump(state, j)),
        syntax::SimpleStatement::Switch(s) => SimpleStatement::Switch(translate_switch(state, s))
    }
}

fn translate_statement(state: &mut State, s: &syntax::Statement) -> Statement {
    match s {
        syntax::Statement::Compound(s) => Statement::Compound(Box::new(translate_compound_statement(state, s))),
        syntax::Statement::Simple(s) => Statement::Simple(Box::new(translate_simple_statement(state, s)))
    }
}


fn translate_compound_statement(state: &mut State, cs: &syntax::CompoundStatement) -> CompoundStatement {
    CompoundStatement { statement_list: cs.statement_list.iter().map(|x| translate_statement(state, x)).collect() }
}

fn translate_function_parameter_declarator(state: &mut State, d: &syntax::FunctionParameterDeclarator) -> FunctionParameterDeclarator {
    FunctionParameterDeclarator {
        ty: d.ty.clone(),
        ident: d.ident.clone(),
    }
}

fn translate_function_parameter_declaration(state: &mut State, p: &syntax::FunctionParameterDeclaration) ->
  FunctionParameterDeclaration
{
    match p {
        syntax::FunctionParameterDeclaration::Named(qual, p) => {
            state.declare(p.ident.ident.as_str(), Type::FullySpecifiedType(
                FullySpecifiedType {
                    qualifier: None,
                    ty: TypeSpecifier {
                        ty: p.ty.ty.clone(),
                        array_specifier: None
                    }
                }));
            FunctionParameterDeclaration::Named(qual.clone(), translate_function_parameter_declarator(state, p))
        }
        syntax::FunctionParameterDeclaration::Unnamed(qual, p) => {
            FunctionParameterDeclaration::Unnamed(qual.clone(), p.clone())
        }

    }
}

fn translate_prototype(state: &mut State, cs: &syntax::FunctionPrototype) -> FunctionPrototype {
    FunctionPrototype {
        ty: cs.ty.clone(),
        name: cs.name.clone(),
        parameters: cs.parameters.iter().map(|x| translate_function_parameter_declaration(state, x)).collect(),
    }
}

fn translate_function_definition(state: &mut State, fd: &syntax::FunctionDefinition) -> FunctionDefinition {
    let prototype = translate_prototype(state, &fd.prototype);
    let params = prototype.parameters.iter().map(|p| match p {
        FunctionParameterDeclaration::Named(_, p) => Type::new(p.ty.ty.clone()),
        FunctionParameterDeclaration::Unnamed(_, p) => Type::new(p.ty.clone()),
    }).collect();
    let sig = FunctionSignature{ ret: Box::new(Type::FullySpecifiedType(prototype.ty.clone())), params };
    state.declare(fd.prototype.name.as_str(), Type::Function(FunctionType{ signatures: NonEmpty::new(sig)}));
    state.push_scope(fd.prototype.name.as_str().into());
    let f = FunctionDefinition {
        prototype,
        statement: translate_compound_statement(state, &fd.statement)
    };
    state.pop_scope();
    f
}

fn translate_external_declaration(state: &mut State, ed: &syntax::ExternalDeclaration) -> ExternalDeclaration {
    match ed {
        syntax::ExternalDeclaration::Declaration(d) =>
            ExternalDeclaration::Declaration(translate_declaration(state, d)),
        syntax::ExternalDeclaration::FunctionDefinition(fd) =>
            ExternalDeclaration::FunctionDefinition(translate_function_definition(state, fd)),
        syntax::ExternalDeclaration::Preprocessor(p) =>
            ExternalDeclaration::Preprocessor(panic!())
    }
}

fn declare_function(state: &mut State, name: &str, ret: Type, params: Vec<Type>) {
    let sig = FunctionSignature{ ret: Box::new(ret), params };
    match state.lookup_sym_mut(name) {
        Some(Symbol { ty: Type::Function(f), ..}) => f.signatures.push(sig),
        None => state.declare(name, Type::Function(FunctionType{ signatures: NonEmpty::new(sig)})),
        _ => panic!("overloaded function name {}", name)
    }
    //state.declare(name, Type::Function(FunctionType{ v}))
}


pub fn ast_to_hir(state: &mut State, tu: &syntax::TranslationUnit) -> TranslationUnit {
    // global scope
    state.push_scope("global".into());
    use TypeSpecifierNonArray::*;
    declare_function(state, "vec3", Type::new(Vec3),
                 vec![Type::new(Float), Type::new(Float), Type::new(Float)]);
    declare_function(state, "vec3", Type::new(Vec3),
                 vec![Type::new(Float)]);
    declare_function(state, "vec3", Type::new(Vec3),
                     vec![Type::new(Vec2), Type::new(Float)]);
    declare_function(state, "vec4", Type::new(Vec4),
                     vec![Type::new(Vec3), Type::new(Float)]);
    declare_function(state, "vec2", Type::new(Vec2),
                     vec![Type::new(Float)]);
    declare_function(state, "mix", Type::new(Vec3),
                 vec![Type::new(Vec3), Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "mix", Type::new(Vec3),
                 vec![Type::new(Vec3), Type::new(Vec3), Type::new(Float)]);
    declare_function(state, "mix", Type::new(Float),
                     vec![Type::new(Float), Type::new(Float), Type::new(Float)]);
    declare_function(state, "step", Type::new(Vec2),
                     vec![Type::new(Vec2), Type::new(Vec2)]);
    declare_function(state, "max", Type::new(Vec2),
                     vec![Type::new(Vec2), Type::new(Vec2)]);
    declare_function(state, "max", Type::new(Float),
                     vec![Type::new(Float), Type::new(Float)]);
    declare_function(state, "min", Type::new(Float),
                     vec![Type::new(Float), Type::new(Float)]);
    declare_function(state, "fwidth", Type::new(Vec2),
                     vec![Type::new(Vec2)]);
    declare_function(state, "clamp", Type::new(Vec3),
                 vec![Type::new(Vec3), Type::new(Float), Type::new(Float)]);
    declare_function(state, "clamp", Type::new(Double),
                     vec![Type::new(Double), Type::new(Double), Type::new(Double)]);
    declare_function(state, "clamp", Type::new(Vec3),
                 vec![Type::new(Vec3), Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "length", Type::new(Float), vec![Type::new(Vec2)]);
    declare_function(state, "pow", Type::new(Vec3), vec![Type::new(Vec3)]);
    declare_function(state, "pow", Type::new(Float), vec![Type::new(Float)]);
    declare_function(state, "lessThanEqual", Type::new(BVec3),
                     vec![Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "if_then_else", Type::new(Vec3),
                     vec![Type::new(BVec3), Type::new(Vec3), Type::new(Vec3)]);
    declare_function(state, "floor", Type::new(Vec4),
                     vec![Type::new(Vec4)]);
    declare_function(state, "floor", Type::new(Double),
                     vec![Type::new(Double)]);
    declare_function(state, "int", Type::new(Int),
                     vec![Type::new(Float)]);
    declare_function(state, "uint", Type::new(UInt),
                     vec![Type::new(Float)]);
    declare_function(state, "uint", Type::new(UInt),
                     vec![Type::new(Int)]);
    declare_function(state, "ivec2", Type::new(IVec2),
                     vec![Type::new(UInt), Type::new(UInt)]);
    declare_function(state, "ivec2", Type::new(IVec2),
                     vec![Type::new(UInt), Type::new(UInt)]);
    declare_function(state, "texelFetch", Type::new(Vec4),
                     vec![Type::new(Sampler2D), Type::new(IVec2), Type::new(Int)]);
    declare_function(state, "texture", Type::new(Vec4),
                     vec![Type::new(Sampler2D), Type::new(Vec3)]);
    state.declare("gl_FragCoord", Type::new(Vec4));


    /*state.declare("clamp", Type::Function(FunctionType { ret: Box::new(Type::Generic) }));
    state.declare("pow", Type::Function(FunctionType { ret: Box::new(Type::Generic) }));
    state.declare("if_then_else", Type::Function(FunctionType { ret: Box::new(Type::Generic) }));
    state.declare("lessThanEqual", Type::Function(FunctionType { ret: Box::new(Type::Generic) }));
*/

    TranslationUnit(tu.0.map(state, translate_external_declaration))
}