# MLIR ODS 핵심 정리

## 머리말
[[밑바닥부터 배우는 딥러닝 컴파일러] 12, MLIR Toy Tutorials 학습 노트 1](https://mp.weixin.qq.com/s/jMHesvKmAUU5dYH0WznulA) 에서 언급했듯이, MLIR은 Dialect를 통해 다양한 수준의 IR을 통일하고, 즉 여러 Operation(연산자)을 정의하는 역할을 한다. 그리고 Dialect와 Operation의 정의는 TableGen 규범을 통해 작성되며, TableGen으로 MLIR의 Operation 정의를 구동하는 방식을 ODS(Operation Definition Specification)라고 부른다. 지금까지는 Toy Tutorials의 Dialect와 Operation이 ODS로 어떻게 정의되는지 간단하게만 살펴봤을 뿐, ODS 자체의 문법이나 여러 제약 조건은 제대로 알지 못했다. 그래서 관련 프로젝트의 Operation 정의를 볼 때마다 어떤 필드가 무슨 의미인지, 또는 사용자 정의 Op를 만들 때 operand나 Attr를 어떻게 선언해야 하는지(예를 들어 Convolution의 groups 파라미터를 optional attribute로 설정하려면 어떻게 해야 하는지) 헷갈리는 일이 많았다.

따라서 이 글에서는 MLIR ODS 문서를 바탕으로 ODS의 핵심 사항들을 설명해, ODS를 더 잘 이해하고 활용할 수 있도록 돕고자 한다. 공식 문서에서 주의해야 할 점들을 작은 단위의 핵심 사항으로 나누어 정리하려 한다. 아래에서 언급하는 TableGen과 ODS는 특별히 구분하지 않는다. ODS의 문법이 곧 TableGen의 문법이기 때문이다. 여기서 소개하는 핵심 사항들은 OneFlow가 MLIR을 연동할 때 모두 어느 정도 사용되었으므로, 관심 있는 분들은 OneFlow의 해당 소스 코드와 비교해 보면 좋다. `https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/include/OneFlow/OneFlowOps.td `.


## 1. 왜 ODS로 Operation을 정의해야 하는가
MLIR에서 Operation을 정의하는 방법은 C++로 직접 정의하는 방식과 ODS 프레임워크 기반으로 정의하는 방식 두 가지가 있다. C++로 직접 정의하려면 베이스 클래스 Op의 일부 생성자를 상속받아 오버라이드해야 하고, Op 하나마다 C++ 코드를 작성해야 한다. 이렇게 하면 시스템 전체의 Op 정의 부분이 매우 장황해지고, 중복 코드가 많이 생기며 가독성도 떨어진다는 것을 쉽게 짐작할 수 있다. 반면 ODS로 Operation을 정의하면 Op 정의를 ODS 규범에 따라 하나의 `td` 파일에 통일해서 작성한 뒤, MLIR이 제공하는 코드 생성 도구로 Operation의 C++ 정의를 자동 생성할 수 있다. 이러한 완전 auto codegen 방식은 Operation 정의를 매우 우아하게 구현하며, 사용자가 신경 써야 할 부분(즉 ODS의 문법 규범)도 훨씬 직관적이다.

ODS는 MLIR에서 Operation을 정의하는 사실상 유일한 선택지이므로, ODS의 문법 규범을 익히는 것이 필수적이다.

## 2. TableGen 문법
TableGen 파일(`.td`로 끝남)에는 다음과 같은 문법 요소들이 포함된다.
- TableGen `class`는 C++의 class와 비슷하며, 템플릿이나 베이스 클래스로 사용해 자식 클래스를 파생시킬 수 있다.
- TableGen `def`는 C++의 객체와 비슷하다. TableGen `class`의 특수화로 선언할 수도 있다. 예를 들어 `def MyDef: MyClass<...>;` 처럼 쓰거나, `def MyDef;` 처럼 단독으로 사용할 수도 있다. 템플릿이나 베이스 클래스로는 사용할 수 없다.
- TableGen `dag`는 directed acyclic graph 요소를 위한 전용 타입이다. `dag` 타입은 하나의 operator와 0개 이상의 인자를 갖는다. 문법은 (`operator arg0, arg1, argN`.) 형태이며, 여기서 `operator`는 임의의 TableGen `def`가 될 수 있다. 인자는 `dag` 자체를 포함해 무엇이든 될 수 있다. (`MyOp:$op_name MyArg:$arg_name`)처럼 operator와 인자에 이름을 붙일 수도 있다.

TableGen에서 지원하는 더 많은 타입과 표현식을 알고 싶다면 다음 링크를 참고한다: https://llvm.org/docs/TableGen/ProgRef.html.

## 3. Operation 정의
MLIR은 Operation 정의를 돕기 위한 몇 가지 공통 구조를 제공하며, 그 의미는 `TableGen backend : OpDefinitionsGen`을 통해 제공된다. 이러한 공통 구조는 `OpBase.td` 파일에 정의되어 있다. 주요 항목은 다음과 같다.

- `Op` 클래스: Operation을 정의할 때 사용하는 주요 구조이다. 이 클래스를 특수화할 때, 아래 구조들의 도움을 받아 Operation과 관련된 모든 사실을 지정한다.
- `Dialect` 클래스: 같은 논리 그룹에 속한 Operation들은 동일한 Dialect 아래에 배치된다. Dialect는 dialect 수준의 정보를 담고 있다.
- `OpTrait` 클래스 및 그 하위 클래스: Operation의 특수 속성과 제약 조건을 지정한다. 예를 들어 Operation에 부수 효과(side effect)가 있는지, Op의 출력이 입력과 동일한 shape를 갖는지 등이다.
- `ins/outs` 마커: `OpDefinitionsGen` 백엔드에 내장된 두 개의 특수 마커로, 각각 operand/attribute와 result의 정의를 안내한다.
- `TypeConstraint` 클래스 및 그 하위 클래스: operand나 result에 대한 제약 조건을 지정한다. 주목할 만한 하위 클래스로는 일반 C++ 타입에 대한 제약을 나타내는 `Type`이 있다.
- `AttrConstraint` 클래스 및 그 하위 클래스: attribute에 대한 제약 조건을 지정한다. 주목할 만한 하위 클래스로는 일반 타입의 값을 갖는 attribute에 대한 제약을 나타내는 `Attr`가 있다.

Operation은 `Op` 클래스를 특수화하여 정의하며, 특수화된 `Op` 클래스에는 필요한 모든 필드의 구체적 내용이 담긴다. 예를 들어, `tf.AvgPool`은 다음과 같이 정의된다.

```cpp
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoSideEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,

    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    Confined<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvertDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
```

아래에서는 Operation 정의에 필요한 모든 필드를 설명한다. 지원되는 필드의 전체 목록은 `Op` 클래스의 정의(즉 `OpBase.td`)를 참고하기 바란다.

- **Operation name**: Operation의 이름이다. 예를 들어 TensorFlow Dialect의 `tf.Add`가 그것이다.
- **Operation documentation**: Operation의 문서 설명으로, `summary`와 `description` 두 가지가 있다. 보면 바로 이해할 수 있는 내용이라 자세한 설명은 생략한다.
- **Operation arguments**: Operation의 인자이다. Operation의 인자는 두 종류가 있는데, 하나는 operand이고 다른 하나는 attribute이다. attribute는 다시 `Natural attributes`와 `Derived attributes`로 나뉜다. 전자는 자연 attribute로 반드시 지정해야 하는 값(예: Convolution의 출력 채널 수)이고, 후자는 파생 attribute로 출력 Tensor의 shape 같은 것이다.

operand와 attribute는 모두 `dag` 타입의 `arguments`에 지정되며, `ins`로 안내된다.

```cpp
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...
  <attr-constraint>:$<attr-name>,
  ...
);
```

여기서 `<type-constraint>`는 `TypeConstraint` 클래스 계층에서 온 TableGen `def`이다. 마찬가지로 `<attr-constraint>`는 `AttrConstraint` 클래스 계층에서 온 TableGen `def`이다. 자세한 내용은 Constraints 절에 있다.


- **가변 operand**. 가변 operand를 정의하려면 `Variadic<...>`로 `TypeConstraint`를 감싸야 한다. 일반적으로 Operation은 가변 operand가 없거나 하나만 있다. 후자의 경우 정적 가변 operand 정의를 통해 동적 가변 operand를 쉽게 추론할 수 있다. 그러나 Operation에 여러 개의 가변 길이 operand(optional이거나 가변 길이인 것)가 있다면, 해당 Operation으로부터 추가 정보가 없는 한 동적 operand를 그에 대응하는 정적 가변 길이 operand 정의에 매핑할 수 없다. 따라서 모든 가변 길이 operand가 그에 대응하는 동적 값을 갖는다는 것을 표시하기 위해 `SameVariadicOperandSize`나 `AttrSizedOperandSegments` trait가 필요하다.
- **optional operand**. optional operand를 정의하려면 `Optional<...>`로 `TypeConstraint`를 감싸야 한다. 설명은 가변 operand와 동일하다.
- **optional attribute**. optional attribute를 정의하려면 `OptionalAttr<...>`로 `AttrConstraint`를 감싸야 한다.
- **기본값을 가진 optional attribute**. `DefaultValuedAttr<..., "...">`로 `AttrConstraint`를 감싼다. `DefaultValuedAttr`의 두 번째 인자는 C++ 기본값을 담은 문자열이어야 한다. 예를 들어 single-precision float 기본값은 `"0.5f"`로, 정수 배열 기본값은 `"{1, 2, 3}"`으로 지정해야 한다.
- **Confining attributes (제약 attribute)**. `Confined`는 값 타입에서 비롯되는 attribute 제약을 추가로 모델링하는 데 도움이 되는 범용 메커니즘으로 제공된다. `Confined`를 사용해 비교적 원시적인 제약을 조합하여 복잡한 제약을 만들 수 있다. 예를 들어, 최솟값이 10인 `32bit` 정수는 `Confined<I32Attr, [IntMinValue<10>]>`로 표현할 수 있다. 다른 예로 `IntMinValue<N>`은 `N` 이상의 정수 attribute를 지정한다는 식이다.
- **Operation results**: operand와 비슷하게, result는 `dag` 타입의 `results`로 선언하며 `outs`로 안내된다.

```cpp
let results = (outs
  <type-constraint>:$<result-name>,
  ...
);
```

- 또한 **Operation regions**과 **Operation successors**가 있는데, 아직 사용해본 적이 없어서 적용 시나리오를 잘 모르겠다.

- **Op의 trait와 constraint (Operation traits and constraints)**: trait는 문법이나 의미에 영향을 주는 Operation의 속성이다. MLIR C++의 다양한 trait는 `mlir::OpTrait` namespace에 있다. Operation의 trait, interface, 또는 여러 operand/attribute/result에 걸친 constraint는 `Op` 클래스의 두 번째 템플릿 인자로 전달해야 한다. 이들은 모두 `OpTrait` 클래스를 상속해야 한다. 자세한 내용은 Constraints 절을 참고한다.

## 4. Operation에 대해 자동 생성되는 기본 builder
Operation을 정의한 뒤에는 어떻게 생성할까? 모든 Operation에 대해 Operation의 인자와 반환값을 기반으로 일부 `builder`가 자동 생성된다. 예를 들어, 다음과 같은 Operation 정의가 주어졌다고 하자.

```cpp
def MyOp : ... {
  let arguments = (ins
    I32:$i32_operand,
    F32:$f32_operand,
    ...,

    I32Attr:$i32_attr,
    F32Attr:$f32_attr,
    ...
  );

  let results = (outs
    I32:$i32_result,
    F32:$f32_result,
    ...
  );
}
```

다음과 같은 `builder`들이 생성된다.

```cpp
// All result-types/operands/attributes have one aggregate parameter.
// 모든 result type/operand/attribute가 하나의 집합 파라미터로 묶인다.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ArrayRef<Type> resultTypes,
                  ValueRange operands,
                  ArrayRef<NamedAttribute> attributes);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are of mlir::Attribute types.
// 각 result type/operand/attribute가 별도의 파라미터이다. attribute 파라미터는 mlir::Attribute 타입이다.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are raw values unwrapped with mlir::Attribute instances.
// (Note that this builder will not always be generated. See the following
// explanation for more details.)
// 각 result type/operand/attribute가 별도의 파라미터이다.
// attribute 파라미터는 mlir::Attribute 인스턴스로 감싸지지 않은 원시 값이다.
// (이 builder는 항상 생성되는 것은 아니다. 자세한 내용은 아래 설명 참고.)
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  APInt i32_attr, StringRef f32_attr, ...);

// Each operand/attribute has a separate parameter but result type is aggregate.
// 각 operand/attribute는 별도의 파라미터이지만, result는 모두 하나의 집합 타입으로 묶인다.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ArrayRef<Type> resultTypes,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...);

// All operands/attributes have aggregate parameters.
// Generated if return type can be inferred.
// 이 builder는 반환 타입이 추론 가능한 경우에만 생성된다.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ValueRange operands, ArrayRef<NamedAttribute> attributes);

// (And manually specified builders depending on the specific op.)
```

위 코드 주석의 번역을 통해 이러한 builder들의 차이점은 이미 설명했다. 이 외에도 다른 builder가 존재할 수 있으니, 자세한 내용은 https://mlir.llvm.org/docs/OpDefinitions/#run-mlir-tblgen-to-see-the-generated-content 의 문서를 참고하기 바란다.



## 5. 사용자 정의 builder
위에서 생성된 C++ 코드의 생성 메서드가 우리가 원하는 것이 아닐 수 있다. 이 경우 사용자 정의 builder를 만들어야 한다. 예를 들면 다음과 같다.

```cpp
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val)>
  ];
}
```

`builders` 필드는 Op 클래스에 추가될 사용자 정의 builder 리스트이다. 위 예제에서는 attribute 대신 float 값을 받는 편의 builder를 제공한다. TableGen `dag`를 사용하는 ODS에서는 많은 함수 선언이 `ins` 접두사를 사용한다. 그 뒤에는 콤마로 구분된 리스트가 이어지며, 리스트의 각 항목은 타입과 `$` 접두사가 붙은 이름의 조합이다. 위 정의는 다음 형식의 builder로 변환된다.


```cpp
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val);
};
```

이 builder에는 두 개의 추가 선행 파라미터가 있다는 점에 주의한다. 이 파라미터들은 Operation을 생성하는 데 유용하다. 특히 이 메서드를 통해 Operation을 생성하려면 `state`에 해당 Operation의 attribute, operand, region, 반환 타입을 채워야 한다. `builder`는 Op에 속하는 임의의 IR 객체(예: 타입이나 중첩된 op)를 생성하는 데 사용할 수 있다. 타입과 이름을 C++ 코드로 변환할 때, 이들은 유효한 C++ 구조여야 한다(Op의 namespace 안의 타입과 식별자, 예를 들어 `class`는 유효한 식별자가 아니다). builder의 구현은 ODS 안에서 직접 제공할 수 있으며, 다음과 같은 TableGen 코드 블록을 사용한다.

```cpp
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

`$_builder`와 `$_state`라는 두 개의 특수 파라미터는 각각 `builder`와 `state`에 해당한다. `ins` 부분의 파라미터는 `val`처럼 직접 사용할 수 있다. builder의 C++ 구현은 ODS 내 특수 변수를 치환하는 방식으로 완성되며, builder ODS 구현의 나머지 부분이 유효한 C++ 구조가 되도록 보장해야 한다. 코드 크기에 제한은 없지만, 짧게 정의되는 builder만 ODS에 인라인으로 두고, 길게 정의되는 builder는 C++ 파일에 두는 것을 권장한다. 마지막으로, 일부 파라미터에 기본값이 필요한 경우 다음과 같이 `CArg`로 타입과 값을 감싸 정의할 수 있다.

```cpp
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins CArg<"float", "0.5f">:$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

변환된 C++ 코드에서는 기본 인자가 선언부에만 나타나고 정의부에는 나타나지 않는데, 이는 C++의 요구사항에 부합한다.

```cpp
/// Header file.
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val = 0.5f);
};

/// Source file.
MyOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
            float val) {
  state.addAttribute("attr", builder.getF32FloatAttr(val));
}
```

## 6. 선언적 어셈블리 포맷 (Declarative Assembly Format)
Operation의 선언적 어셈블리 포맷은 Operation의 operand, attribute 등과 매칭되는 선언적 문자열로 지정할 수 있다. Operation을 파싱해 구성하는 데 필요한 추가 정보를 표현할 수 있는 능력도 갖추고 있다.


```cpp
def CallOp : Std_Op<"call", ...> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}
```

크게 세 부분으로 구성된다.

- **Directives (지시자)**. directive는 옵션 인자를 받는 빌트인 함수이다. 사용 가능한 directive에는 `attr-dict`, `attr-dict-with-keyword`, `operands`, `ref` 등이 있다.
- **리터럴 (Literals)**. 리터럴은 백틱으로 감싼 키워드나 구두점이다. 다음은 유효한 구두점 집합이다: `:, ,, =, <, >, (, ), {, }, [, ], ->, ?, +, *`. `\n` 구두점은 줄바꿈 효과를 갖는다. 다음과 같다.

```cpp
let assemblyFormat = [{
  `{` `\n` ` ` ` ` `this_is_on_a_newline` `\n` `}` attr-dict
}];
```

```cpp
%results = my.operation {
  this_is_on_a_newline
}
```
내용이 비어 있는 리터럴은 특정 리터럴 요소가 묵시적으로 삽입한 공백을 제거하는 데 사용할 수 있다. 예를 들어 `)`나 `]` 등이다. 예를 들어 `]`가 output의 끝에 나타날 수 있지만 포맷의 마지막 요소가 아닐 때, 이 경우 `"]``"`를 사용해 뒤따르는 공백을 제거할 수 있다.

- **변수 (Variables)**. 변수는 Operation에 등록된 엔티티로, 예를 들어 Operation의 인자(attribute나 operand), region, result, successor 등이다. `CallOp`에서 변수는 `$callee`와 `$args`를 가리킨다. attribute 변수는 그 값의 타입을 함께 표시한다. 단, 그 값의 타입이 구성 가능한 경우에는 attribute 변수의 값 타입을 생략할 수 있다.


## 7. Custom Directives & Optional Groups
선언적 어셈블리 포맷 사양은 Operation을 포맷팅할 때 대부분의 일반적인 시나리오를 처리할 수 있다. 포맷 안에서 Operation의 특정 부분을 지정하고 싶지만 선언적 문법이 지원하지 않는 경우, custom directive를 사용해볼 수 있다.

어떤 경우 Operation은 "선택적인" 정보를 가질 수 있다. 예를 들어 attribute나 비어 있는 가변 길이 operand 그룹 같은 것이다. 이런 경우 해당 정보의 존재 여부에 따라 어셈블리 포맷의 일부를 optional로 표시할 수 있다.

이 두 부분은 다소 복잡하고 아직 사용해보지 않았으므로, 여기서는 자세히 설명하지 않는다. 관심 있는 분들은 공식 문서를 참고하기 바란다.

## 8. 타입 추론
포맷의 한 가지 요구사항은 operand와 result의 타입이 항상 존재해야 한다는 것이다. 어떤 경우 type constraint나 다른 가용 정보를 통해 변수의 타입을 추론할 수 있다. 이런 경우 포맷에서 해당 변수의 타입을 생략할 수 있다.
- **Buildable Types (구성 가능한 타입)**. 일부 type constraint는 표현 방식이 하나뿐이어서 직접 구성할 수 있다. 예를 들어 `I32`나 `Index` 타입이 그렇다. ODS에서 타입은 `builderCall` 필드를 설정하거나 `BuildableType` 클래스를 상속받아 자신을 구성 가능한 타입으로 표시할 수 있다.
- **Trait Equality Constraints (Trait 동등 제약)**. 많은 Operation들은 Operation에 알려진 타입 동등 trait로 등록된 제약을 갖는다. 예를 들어 `select` Operation의 true, false, result 값은 보통 동일한 타입이다. 어셈블리 포맷은 이런 동등 제약들을 검사해 누락된 변수의 타입을 알아낼 수 있다. 현재 지원되는 trait에는 `AllTypesMatch`, `TypesMatchWith`, `SameTypeOperands`, `SameOperandsAndResultType`이 있다.
- **InferTypeOpInterface**. `InferTypeOpInterface`를 구현한 Operation은 어셈블리 포맷에서 result 타입을 생략할 수 있다. operand로부터 result 타입을 추론할 수 있기 때문이다.
- **hasCanonicalizer**. 이 boolean 필드는 해당 Operation에 대한 canonicalization 패턴이 정의되었는지 여부를 나타낸다. `1`이면 `::getCanonicalizationPatterns()`가 정의되어야 한다.
- **hasCanonicalizeMethod**. 이 boolean 필드가 `true`로 설정되면, Operation이 단순한 "matchAndRewrite" 스타일의 canonicalization 패턴을 위해 `canonicalize` 메서드를 구현했음을 나타낸다. `hasCanonicalizer`가 0이면, 이 함수를 호출하기 위한 `::getCanonicalizationPatterns()`의 구현이 자동으로 만들어진다.
- **hasFolder**. 이 boolean 필드는 해당 Operation에 대한 일반 folding 규칙이 정의되었는지 여부를 나타낸다. `1`이면 `::fold()`가 정의되어야 한다.

## 9. 추가 선언
표 기반(table-driven) Operation 정의의 목표 중 하나는 각 Operation에 대해 가능한 한 많은 로직과 메서드를 자동 생성하는 것이다. 그렇긴 해도 항상 처리할 수 없는 long tail 케이스는 존재한다. 이런 경우에는 `extraClassDeclaration`을 사용할 수 있다. `extraClassDeclaration`의 코드는 생성된 C++ op 클래스에 그대로 복사된다.

`extraClassDeclaration`은 고급 사용자를 위한 long tail 케이스 메커니즘이라는 점에 유의하자. 아직 구현되지 않은 광범위하게 적용 가능한 케이스의 경우, 인프라 자체를 개선하는 편이 더 바람직하다.

## 10. C++ 코드 생성
`OpDefinitionsGen` (https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp)은 Operation 정의 사양 파일(`.td` 파일)을 처리해 두 개의 C++ 코드 파일을 생성한다: 하나는 선언용, 다른 하나는 정의용이다. 전자는 `-gen-op-decls` 명령행 옵션으로 생성되고, 후자는 `-gen-op-defs` 옵션으로 생성된다.

정의 파일에는 모든 op의 메서드 정의가 들어 있으며, `GET_OP_CLASSES`를 정의해 포함하고 활성화할 수 있다. 각 Operation에 대해 OpDefinitionsGen은 operation 클래스와 operand adaptor 클래스를 생성한다. 또한 정의된 모든 Operation을 콤마로 구분한 리스트도 포함하는데, `GET_OP_LIST`를 정의해 포함하고 활성화할 수 있다.

- **클래스 이름과 namespace**.

각 Operation에 대해 생성되는 C++ 클래스 이름은 TableGen `def`를 접두사로 한 이름에서 Dialect 접두사를 제거한 형태이다. 첫 번째 `_`가 구분자로 사용된다. 예를 들어 `def TF_AddOp`의 경우 C++ 클래스 이름은 `AddOp`가 된다. `TF` 접두사는 여러 Operation의 scope이므로 제거된다. 다른 Dialect도 자체 AddOps를 정의할 수 있다.

생성된 C++ 클래스의 namespace는 Dialect의 `cppNamespace` 필드에서 가져온다. 예를 들어, Dialect의 `Namespace`가 `A::B`라면, 그 Dialect의 Op는 `namespace A { namespace B { ... } }`에 배치된다. Dialect가 `cppNamespace`를 지정하지 않으면, dialect 이름을 namespace로 사용한다.

이는 생성된 C++ 클래스의 이름이 Operation 이름의 op 이름과 반드시 정확히 일치하지는 않음을 의미한다. 이는 코딩 스타일 요구사항을 충족하기 위한 유연한 명명을 허용하기 위함이다.


- **Operand adaptors**

각 Operation에 대해 MLIR은 operand adaptor를 자동 생성한다. 이 클래스는 list 값으로 제공되는 operand들을 "매직" 상수 없이 접근할 수 있도록 해준다. operand adaptor는 `Value` 배열을 참조하며, Operation 클래스와 동일한 이름의 메서드들을 제공해 그 값들에 접근할 수 있게 한다. 예를 들어 이항 산술 연산이라면 첫 번째 operand에 접근하기 위한 `.lhs()`와 두 번째 operand에 접근하기 위한 `.rhs()`를 제공할 수 있다. operand adaptor 클래스는 Operation 클래스와 같은 namespace에 위치하며, 클래스 이름은 Operation 클래스 이름 뒤에 `Adaptor`를 붙인 형태이다.

operand adaptor는 Operation을 다루는 함수 템플릿에서도 사용할 수 있다.

```cpp
template <typename BinaryOpTy>
std::pair<Value, Value> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());;
}

void process(AddOp op, ArrayRef<Value> newOperands) {
  zip(op);
  zip(Adaptor<AddOp>(newOperands));
  /*...*/
}

```

OneFlow에서는 생성된 `UserOpAdaptor` 코드를 볼 수 있다. 그 안에는 Operation의 operand와 관련 attribute에 접근할 수 있는 일련의 인터페이스가 제공되어 있다.

```cpp
//===----------------------------------------------------------------------===//
// ::mlir::oneflow::UserOp declarations
//===----------------------------------------------------------------------===//

class UserOpAdaptor {
public:
  UserOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});
  UserOpAdaptor(UserOp &op);
  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange data_input();
  ::mlir::ValueRange ctrl_inputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr op_name();
  ::mlir::BoolAttr trainable();
  ::mlir::StringAttr device_tag();
  ::mlir::ArrayAttr device_name();
  ::mlir::IntegerAttr scope_symbol_id();
  ::mlir::ArrayAttr hierarchy();
  ::mlir::DenseIntElementsAttr operand_segment_sizes();
  ::mlir::DenseIntElementsAttr result_segment_sizes();
  ::mlir::StringAttr op_type_name();
  ::mlir::ArrayAttr input_lbn_segment_keys();
  ::mlir::ArrayAttr input_lbn_segment_sizes();
  ::mlir::ArrayAttr output_lbn_segment_keys();
  ::mlir::ArrayAttr output_lbn_segment_sizes();
  ::mlir::ArrayAttr output_lbns();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
```

## 11. Constraint
Constraint는 표 기반 Operation 정의의 핵심 개념이다. Operation 검증과 그래프 Operation 매칭은 모두 constraint 충족 여부를 기반으로 이루어진다. 따라서 Operation 정의와 rewrite 규칙 모두 constraint 작성과 직접 관련된다. MLIR은 `OpBase.td`(`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`)에 `Constraint` 베이스 클래스를 정의한다. Operation의 constraint는 다음과 같은 다양한 범위를 다룰 수 있다.

- 단일 attribute에만 관련 (예: 5보다 큰 32비트 정수)
- 여러 operand와 result (예: 첫 번째 result의 shape는 첫 번째 operand(Tensor로 이해할 수 있다)와 동일해야 함)
- Operation 자체에 내재된 것 (예: 부수 효과 없음, Transpose Op 제거 케이스 참고)

이들을 각각 single-entity constraint, multi-entity constraint, trait이라고 부른다.



## 머리말
이 절에서는 [[밑바닥부터 배우는 딥러닝 컴파일러] 16, MLIR ODS 핵심 정리 상편](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 을 바탕으로 ODS의 핵심 사항들을 보충하여 완성한다. constraint와 attribute의 정의는 모두 MLIR에서 매우 중요한 요소이며, 타입 정의는 개인적으로는 알고만 있어도 충분하다고 본다. 사용자 정의 타입이 필요한 시점에 자세히 연구하면 된다. 마지막으로 MLIR 문법은 다소 난해한데, 입문자는 `mlir-tblgen`을 활용해 디버깅에 도움을 받을 수 있다.

이 두 편의 글에서 나는 MLIR ODS 사양을 한 번 완주하면서 14개의 핵심 사항을 정리했고, 각 핵심 사항마다 OneFlow MLIR의 Op 정의에서 대조해보고 예제 코드와 위치를 함께 제시했다. 독자들의 MLIR 입문에 도움이 되기를 바란다.

## 11. Constraint (이건 매우 중요)
Constraint는 표 기반 Operation 정의의 핵심 개념이다. Operation 검증과 그래프 Operation 매칭은 모두 constraint를 기반으로 이루어진다. 따라서 Operation 정의와 rewrite 규칙 모두 constraint 작성과 직접 관련된다. MLIR은 `OpBase.td`(`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`)에 `Constraint` 베이스 클래스를 정의한다. Operation의 constraint는 다음과 같은 다양한 범위를 다룰 수 있다.

- 단일 attribute에만 관련 (예: 5보다 큰 32비트 정수)
- 여러 operand와 result (예: 첫 번째 result의 shape는 첫 번째 operand(Tensor로 이해할 수 있다)와 동일해야 함)
- Operation 자체에 내재된 것 (예: 부수 효과 없음, Transpose Op 제거 케이스 참고)

이들을 각각 single-entity constraint, multi-entity constraint, trait이라고 부른다. 이 부분의 개념은 알고만 있으면 되며, 새로운 constraint를 작성하는 것이 가장 중요하다고 생각한다.

- **single-entity constraint**. single-entity constraint의 적용 범위는 단일 operand, attribute, 또는 result이며, constraint는 엔티티가 선언된 위치에서 지정한다. 예를 들어 **Operation arguments**와 **Operation results**가 그것이다([[밑바닥부터 배우는 딥러닝 컴파일러] 16, MLIR ODS 핵심 정리 상편](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 에서 Operation arguments와 Operation results와 관련해 주의해야 할 지식들을 정리했다).
- **multi-entity constraint**. multi-entity constraint는 `https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`에서 `PredOpTrait` 클래스(`OpTrait`의 하위 클래스)로 모델링된다. 전체 목록은 `OpBase.td`를 참고한다.
- **trait**. trait은 Operation의 내재 속성이다. 예를 들어 부수 효과 유무, 교환 가능 여부, terminator 여부 등이다. 이러한 constraint들은 Op 클래스 템플릿 인자로 지정해야 하며, 이는 [[밑바닥부터 배우는 딥러닝 컴파일러] 16, MLIR ODS 핵심 정리 상편](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 의 3절 "Op의 trait와 constraint (Operation traits and constraints)"에서 보여준 바와 같다. trait은 `https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`에서 `NativeOpTrait` 클래스(`OpTrait`의 하위 클래스)로 모델링된다. 이들은 지원되며, 그에 대응하는 C++ `mlir::OpTrait` 클래스로 변환된다.

- **새 constraint는 어떻게 지정하는가?** 새 constraint를 작성하려면 그것에 대해 술어(predicate)를 제공하고 설명용 이름을 지정해야 한다. `Pred` 클래스로 모델링되는 predicate는 constraint를 구성하는 핵심이다. constraint의 predicate는 보통 중첩된 방식으로 구성되며, 두 종류의 predicate가 있다: 1. `CPred`: 원시 leaf 노드 predicate. 2. 복합 predicate: predicate 결합자를 사용해 하위 predicate들을 조합한 predicate (conjunction: `And`, disjunction: `Or`, negation: `Neg`, substitution: `SubstLeaves`, concatenation: `Concat`). `CPred`는 더 복잡한 predicate를 구성하는 기초이다. 이는 TableGen 관점에서의 "원자적" predicate로서, TableGen과 C++ 사이의 "인터페이스" 역할을 한다. 그 안은 이미 C++ 코드이며, 특수 자리표시자가 치환될 불투명 문자열로 취급된다. boolean을 반환하는 어떤 C++ 코드든 `CPred` 안에 넣을 수 있는데, 표현식 계산, 함수 호출, 클래스 메서드 호출 등이 모두 가능하다.

C++ 환경과의 상호작용을 돕기 위해, predicate가 사용되는 context의 엔티티를 참조할 수 있는 몇 가지 특수 자리표시자가 제공된다. 이들은 둘러싼 환경에 대한 "후크" 역할을 한다. `$_builder`, `$_op`, `$_self`가 그것이다.

- `$_builder`는 `mlir::Builder` 인스턴스로 치환되어, 일반적인 builder 메서드에 접근할 수 있게 해준다.
- `$_op`은 현재 Operation으로 치환되어, 현재 Operation의 정보에 접근할 수 있게 해준다.
- `$_self`는 그 predicate가 부착된 엔티티로 치환된다. 예를 들어 `BoolAttr`는 `CPred<"$_self.isa<BoolAttr>()">`을 포함하는 attribute constraint이다. 이때 `BoolAttr:$attr`의 경우, `$_self`는 `$attr`로 치환된다. type constraint의 경우는 약간 특별한데, 각 타입 정의의 constraint가 자연스럽게 읽히길 원하고 type constraint를 operand/result에 직접 부착하길 원하기 때문에, `$_self`는 operand/result의 타입으로 치환된다. 예를 들어 `F32:$operand`에서의 `F32`의 경우, `$_self`는 `operand(...).getType()`으로 확장된다.

예를 들어, attribute `attr`이 `IntegerAttr`인지 작성하려 할 때 C++에서는 `attr.isa<IntegerAttr>()`로 호출해 구현할 수 있다. 이 코드는 `$_self.isa<IntegerAttr>()` 형태로 `CPred`에 감싸 사용할 수도 있는데, 여기서 `$_self`는 특수 자리표시자로서 확장 시 현재 attribute `attr`로 치환되어 동일한 기능을 (Tablegen에서) 구현하게 된다.

더 복잡한 predicate의 경우, 단일 `CPred`로 감싸거나 predicate 결합자로 조합할 수 있다. 예를 들어, attribute `attr`이 32비트 또는 64비트 정수라는 constraint를 작성하면 다음과 같이 쓸 수 있다.

```cpp
And<[
  CPred<"$_self.isa<IntegerAttr>()">,
  Or<[
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(32)">,
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(64)">
  ]>
]>
```
(참고로 위는 `CPred`와 predicate 결합자를 사용해 복잡한 predicate를 작성하는 방법을 익숙한 예로 보여준 것일 뿐이다. 구체적으로 정수 attribute에 대해서는 `OpBase.td`에 이미 `I32Attr`와 `I64Attr`가 정의되어 있다. 따라서 실제로는 이를 재사용해 `Or<[I32Attr.predicate, I64Attr.predicate]>`로 작성할 수 있다.)

여기서 OneFlow의 예를 하나 더 들어 설명한다. 우리는 IsGPU라는 constraint를 정의했다.

```cpp
def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;
```

그리고 OneFlow는 Transformer 부분에서 맞춤 최적화를 적용했는데, 바로 Scale과 Tril이라는 두 개의 연속된 Kernel을 하나의 큰 Kernel로 fuse하여 메모리 read/write 시간을 일부 절약하는 것이다. 그런데 이 fuse된 kernel은 GPU에서만 적용되므로, 이때 현재 계산 그래프에서 검출된 Scale과 Tril 두 Operation의 device가 GPU인지 판별해야 하며, 따라서 이 constraint가 필요하다. FusedScaleTrilPattern Pass의 구현은 다음과 같으며, 마지막에 IsGPU constraint가 사용된 것을 볼 수 있다.

```cpp
def FusedScaleTrilPattern : Pat<
  (
    OneFlow_TrilOp
    (
      OneFlow_ScalarMulOp
        $x,
        $scale_op_name,
        $scale_trainable,
        $scale_device_tag,
        $scale_device_name,
        $scale_scope_symbol_id,
        $scale_hierarchy,
        $has_int_operand,
        $has_float_operand,
        $int_operand,
        $float_operand
    ),
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value
  ),
  (OneFlow_FusedScaleTrilOp $x,
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value,
    $float_operand,
    $int_operand,
    $has_float_operand
  ),
  [
    (IsGPU $tril_device_tag),
    (IsGPU $scale_device_tag)
  ]
>;
```

이 Pass의 기능은 연속된 Scale+Tril Operation을 검출하면 이 두 Operation을 하나의 FusedScaleTril Operation으로 fuse하는 것이다.

만약 predicate를 `CPred`와 predicate 결합자로 작성하는 것이 너무 복잡하다면, 이를 일반 C++ 함수로 작성하고 `CPred`를 그 함수를 "호출"하는 수단으로 사용할 수도 있다. 예를 들어, attribute `attr`이 어떤 속성을 갖는지 검증하려면 다음과 같은 C++ 함수를 작성할 수 있다.

```cpp
bool HasSomeProperty(Attribute attr) { ... }
```

그런 다음 Op를 다음과 같이 정의한다.

```cpp
def HasSomeProperty : AttrConstraint<CPred<"HasSomeProperty($_self)">,
                                     "has some property">;

def MyOp : Op<...> {
  let arguments = (ins
    ...
    HasSomeProperty:$attr
  );
}
```

predicate를 정의할 때 단일 `CPred`로 전체 표현식을 감쌀지, predicate 결합자가 들어간 여러 `CPred`로 작성할지, 아니면 단일 `CPred`로 함수를 "호출"할지에 대한 명확한 기준은 없다. `CPred`와 predicate 결합자를 사용해 정의하는 편이 바람직한데, 이는 더 많은 정보를 (C++ 함수 뒤에 모든 로직을 숨기는 대신) operation 정의 사양에 노출시켜, 잠재적으로 더 많은 자동 생성 케이스를 구동할 수 있게 하기 때문이다. 다만 이를 위해서는 빌딩 블록으로 쓸 잘 정리된 범용 predicate 라이브러리가 필요하며, 중복을 피하기 위해 현재 연구가 진행 중이다.

## 12. Attribute 정의 (이것도 매우 중요 +1)
attribute는 컴파일 타임에 알 수 있는 Operation의 상수이다. ODS는 C++ attribute 클래스 위에 attribute wrapper를 제공한다. MLIR의 코어 IR 라이브러리에는 일부 일반적인 C++ attribute 클래스가 정의되어 있다(`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Attributes.h`). ODS는 이런 attribute들을 TableGen에서 사용해 더 세분화된 constraint를 곁들여 Operation을 정의할 수 있게 해준다. 예를 들어 `StrAttr`는 `StringAttr`에 직접 매핑되며, `F32Attr/F64Attr`은 `FloatAttr`이 추가로 특정 비트 폭을 가질 것을 요구한다. ODS attribute는 storage 타입(attribute를 저장하는 `mlir::Attribute` 클래스에 해당), return 타입(생성된 getter 헬퍼 함수의 C++ 반환 타입에 해당), 그리고 내부 storage 타입과 헬퍼 함수 사이를 변환하는 메서드를 갖도록 정의된다.

**attribute 데코레이터**. ODS attribute에 적용하여 optionality, 기본값 등 일반적인 추가 속성을 지정할 수 있는 중요한 attribute adaptor/decorator/modifier들이 있다.
- `DefaultValuedAttr`: attribute에 기본값을 지정한다.
- `OptionalAttr`: attribute를 optional로 지정한다.
- `Confined`: `Confined`는 값 타입에서 비롯되는 attribute 제약을 추가로 모델링하는 데 도움이 되는 범용 메커니즘으로 제공된다. `Confined`를 사용해 비교적 원시적인 제약을 조합하여 복잡한 제약을 만들 수 있다. 예를 들어, 최솟값이 10인 `32bit` 정수는 `Confined<I32Attr, [IntMinValue<10>]>`로 표현할 수 있다. 다른 예로 `IntMinValue<N>`은 N 이상의 정수 attribute를 지정한다는 식이다.

**enum attribute**. 어떤 attribute는 사전에 정의된 enum에서만 값을 가질 수 있다. 예를 들어 비교 op의 비교 종류 같은 것이다. 이러한 attribute를 정의하기 위해 ODS는 몇 가지 메커니즘을 제공한다: `StrEnumAttr`, `IntEnumAttr`, `BitEnumAttr`.
- `StrEnumAttr`: 각 enum case가 문자열이며, attribute는 op에 `StringAttr`로 저장된다.
- `IntEnumAttr`: 각 enum case가 정수이며, attribute는 op에 `IntegerType`으로 저장된다.
- `BitEnumAttr`: 각 enum case가 비트이며, attribute는 op에 `IntegerAttr`로 저장된다.

이 모든 `*EnumAttr` attribute는 그에 대응하는 `*EnumAttrCase`를 통해 허용되는 모든 케이스를 완전히 지정해야 한다. 이를 통해 ODS는 허용된 케이스만 받아들이도록 추가 검증을 생성할 수 있다. `*EnumAttr`와 그 C++ 사용자 사이의 상호작용을 촉진하기 위해, EnumsGen(`https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/EnumsGen.cpp`) TableGen 백엔드는 몇 가지 일반적인 유틸리티를 생성할 수 있다: C++ enum 클래스, enum 클래스용 `llvm::DenseMapInfo`, 문자열로/문자열에서의 변환 함수 등이다. 이는 `mlir-tblgen`의 `-gen-enum-decls`와 `-gen-enum-defs` 명령행 옵션으로 제어된다.

예를 들어, 다음 `EnumAttr`가 주어졌을 때:



```cpp
def Case15: I32EnumAttrCase<"Case15", 15>;
def Case20: I32EnumAttrCase<"Case20", 20>;

def MyIntEnum: I32EnumAttr<"MyIntEnum", "An example int enum",
                           [Case15, Case20]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
```
다음 코드가 `mlir-tblgen -gen-enum-decls`로 생성된다.

```cpp
namespace Outer {
namespace Inner {
// An example int enum
enum class MyIntEnum : uint32_t {
  Case15 = 15,
  Case20 = 20,
};

llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t);
llvm::StringRef ConvertToString(MyIntEnum);
llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMyIntEnum() {
  return 20;
}

} // namespace Inner
} // namespace Outer

namespace llvm {
template<> struct DenseMapInfo<Outer::Inner::MyIntEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline Outer::Inner::MyIntEnum getEmptyKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getEmptyKey());
  }

  static inline Outer::Inner::MyIntEnum getTombstoneKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Outer::Inner::MyIntEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const Outer::Inner::MyIntEnum &lhs, const Outer::Inner::MyIntEnum &rhs) {
    return lhs == rhs;
  }
};
}
```

다음 코드는 `mlir-tblgen -gen-enum-defs`로 생성된다.


```cpp
namespace Outer {
namespace Inner {
llvm::StringRef ConvertToString(MyIntEnum val) {
  switch (val) {
    case MyIntEnum::Case15: return "Case15";
    case MyIntEnum::Case20: return "Case20";
  }
  return "";
}

llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef str) {
  return llvm::StringSwitch<llvm::Optional<MyIntEnum>>(str)
      .Case("Case15", MyIntEnum::Case15)
      .Case("Case20", MyIntEnum::Case20)
      .Default(llvm::None);
}
llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t value) {
  switch (value) {
  case 15: return MyIntEnum::Case15;
  case 20: return MyIntEnum::Case20;
  default: return llvm::None;
  }
}

} // namespace Inner
} // namespace Outer
```

다음 `BitEnumAttr` 정의에 대해서도 비슷하다.

```cpp
def None: BitEnumAttrCase<"None", 0x0000>;
def Bit1: BitEnumAttrCase<"Bit1", 0x0001>;
def Bit2: BitEnumAttrCase<"Bit2", 0x0002>;
def Bit3: BitEnumAttrCase<"Bit3", 0x0004>;

def MyBitEnum: BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [None, Bit1, Bit2, Bit3]>;
```

다음을 얻는다.

```cpp
// An example bit enum
enum class MyBitEnum : uint32_t {
  None = 0,
  Bit1 = 1,
  Bit2 = 2,
  Bit3 = 4,
};

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t);
std::string stringifyMyBitEnum(MyBitEnum);
llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef);
inline MyBitEnum operator|(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline MyBitEnum operator&(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(MyBitEnum bits, MyBitEnum bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

namespace llvm {
template<> struct DenseMapInfo<::MyBitEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline ::MyBitEnum getEmptyKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getEmptyKey());
  }

  static inline ::MyBitEnum getTombstoneKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::MyBitEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::MyBitEnum &lhs, const ::MyBitEnum &rhs) {
    return lhs == rhs;
  }
};
```

```cpp
std::string stringifyMyBitEnum(MyBitEnum symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  llvm::SmallVector<llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Bit1"); val &= ~1u; }
  if (2u & val) { strs.push_back("Bit2"); val &= ~2u; }
  if (4u & val) { strs.push_back("Bit3"); val &= ~4u; }

  if (val) return "";
  return llvm::join(strs, "|");
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return MyBitEnum::None;

  llvm::SmallVector<llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<llvm::Optional<uint32_t>>(symbol)
      .Case("Bit1", 1)
      .Case("Bit2", 2)
      .Case("Bit3", 4)
      .Default(llvm::None);
    if (bit) { val |= *bit; } else { return llvm::None; }
  }
  return static_cast<MyBitEnum>(val);
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return MyBitEnum::None;

  if (value & ~(1u | 2u | 4u)) return llvm::None;
  return static_cast<MyBitEnum>(value);
}
```

OneFlow-MLIR에도 OneFlow의 다양한 데이터 타입을 다루기 위한 enum attribute 정의가 있다. 코드는 다음과 같다.

```cpp
#ifndef ONEFLOW_ENUMS
#define ONEFLOW_ENUMS

def OneFlow_InvalidDataType : I32EnumAttrCase<"DT_InvalidDataType", 0>;
def OneFlow_Char : I32EnumAttrCase<"DT_Char", 1>;
def OneFlow_Float : I32EnumAttrCase<"DT_Float", 2>;
def OneFlow_Double : I32EnumAttrCase<"DT_Double", 3>;
def OneFlow_Int8 : I32EnumAttrCase<"DT_Int8", 4>;
def OneFlow_Int32 : I32EnumAttrCase<"DT_Int32", 5>;
def OneFlow_Int64 : I32EnumAttrCase<"DT_Int64", 6>;
def OneFlow_UInt8 : I32EnumAttrCase<"DT_UInt8", 7>;
def OneFlow_OFRecord : I32EnumAttrCase<"DT_OFRecord", 8>;
def OneFlow_Float16 : I32EnumAttrCase<"DT_Float16", 9>;
def OneFlow_TensorBuffer: I32EnumAttrCase<"DT_TensorBuffer", 10>;

def OneFlow_DataType: I32EnumAttr<"DataType", "OneFlow Data Type enum",
  [
    OneFlow_InvalidDataType,
    OneFlow_Char,
    OneFlow_Float,
    OneFlow_Double,
    OneFlow_Int8,
    OneFlow_Int32,
    OneFlow_Int64,
    OneFlow_UInt8,
    OneFlow_OFRecord,
    OneFlow_Float16,
    OneFlow_TensorBuffer,
  ]
> {
  let cppNamespace = "::mlir::oneflow";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

#endif // ONEFLOW_ENUMS
```

이로부터 생성되는 enum attribute 선언을 살펴보자.


```cpp
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace oneflow {
// OneFlow Data Type enum
enum class DataType : uint32_t {
  DT_InvalidDataType = 0,
  DT_Char = 1,
  DT_Float = 2,
  DT_Double = 3,
  DT_Int8 = 4,
  DT_Int32 = 5,
  DT_Int64 = 6,
  DT_UInt8 = 7,
  DT_OFRecord = 8,
  DT_Float16 = 9,
  DT_TensorBuffer = 10,
};

::llvm::Optional<DataType> symbolizeDataType(uint32_t);
::llvm::StringRef ConvertToString(DataType);
::llvm::Optional<DataType> ConvertToEnum(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDataType() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(DataType enumValue) {
  return ConvertToString(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DataType> symbolizeEnum<DataType>(::llvm::StringRef str) {
  return ConvertToEnum(str);
}

class DataTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DataType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DataTypeAttr get(::mlir::MLIRContext *context, DataType val);
  DataType getValue() const;
};
} // namespace oneflow
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::oneflow::DataType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::oneflow::DataType getEmptyKey() {
    return static_cast<::mlir::oneflow::DataType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::oneflow::DataType getTombstoneKey() {
    return static_cast<::mlir::oneflow::DataType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::oneflow::DataType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::oneflow::DataType &lhs, const ::mlir::oneflow::DataType &rhs) {
    return lhs == rhs;
  }
};
}
```

구현 부분은 코드가 너무 길어 여기서는 붙이지 않는다.

## 13. 타입 정의 (간단히 알아보는 정도만)
MLIR은 그 사양에 따라 데이터 타입을 생성할 수 있도록 `TypeDef` 클래스 계층을 정의한다. 타입은 필요한 모든 필드의 구체적인 내용을 갖는 `TypeDef` 클래스를 특수화하여 정의한다. 예를 들어 정수 타입은 다음과 같이 정의할 수 있다.


```cpp
// All of the types will extend this class.
class Test_Type<string name> : TypeDef<Test_Dialect, name> { }

// An alternate int type.
def IntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";

  let summary = "An integer type with special semantics";

  let description = [{
    An alternate integer type. This type differentiates itself from the
    standard integer type by not having a SignednessSemantics parameter, just
    a width.
  }];

  let parameters = (ins "unsigned":$width);

  // We define the printer inline.
  let printer = [{
    $_printer << "int<" << getImpl()->width << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width);
  }];
}
```

- **Type name**: 생성된 C++ 클래스의 이름은 기본적으로 `<classParamName>Type`이다(위 예제의 `TestIntegerType`). 이는 `cppClassName` 필드로 재정의할 수 있다. `mnemonic`은 파싱용 asm 이름을 지정한다. 이는 optional이며, 지정하지 않으면 이 클래스에 parser나 printer 메서드가 부착되지 않음을 의미한다.
- **Type documentation**: `summary`와 `description` 필드가 있으며, 사용 방식은 Operation에서와 동일하다. 즉 `summary`는 한 줄이어야 하고, `description`은 더 긴 설명이어야 한다.
- **Type parameters**: `parameters` 필드는 타입 파라미터의 리스트이다. 파라미터를 지정하지 않으면(기본값) 이 타입은 singleton 타입으로 간주된다. 파라미터는 `"c++Type":$paramName` 형식이다. storage 생성자에서 할당이 필요한 C++ 타입을 파라미터로 사용하려면 두 가지 옵션이 있다: 1. `hasCustomStorageConstructor`를 설정하여 선언만 있는 생성자를 갖는 TypeStorage 클래스를 생성하게 한 뒤, 정의는 직접 작성한다. 2. "c++Type" 문자열 대신 `TypeParameter` tablegen 클래스를 사용한다. (뒷부분의 표현은 잘 이해가 되지 않고 사용해본 적도 없다.)

- **TypeParameter tablegen class**: 이는 각 타입 파라미터에 관한 속성을 추가로 지정하는 데 사용된다. 여기에는 문서(`summary`와 `syntax`), 사용할 C++ 타입, storage 생성자 메서드에서 사용할 사용자 정의 할당자, 그리고 파라미터 타입의 두 인스턴스가 동일한지를 판별하는 사용자 정의 비교자가 포함된다.

```cpp
// DO NOT DO THIS!
let parameters = (ins "ArrayRef<int>":$dims);
```
기본 storage 생성자는 필드를 무작정 값으로 복사한다. 타입에 대해서는 아무것도 알지 못한다. 이 경우 ArrayRef는 `dims = allocator.copyInto(dims)`로 할당해 주어야 한다.

```cpp
class ArrayRefIntParam :
    TypeParameter<"::llvm::ArrayRef<int>", "Array of ints"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

...

let parameters = (ins ArrayRefIntParam:$dims);
```
`allocator` 코드 블록은 `$_allocator`(객체가 할당될 TypeStorageAllocator)와 `$_dst`(할당된 데이터를 담을 변수)로 구성된다. `comparator` 코드 블록은 `$_lhs`와 `$_rhs`라는 파라미터 타입 인스턴스로 구성된다.

사용자 정의 Type에는 더 많은 내용이 있지만, 현재로서는 그쪽 수요가 없으므로 더 보지 않고 여기까지만 간단히 살펴보았다. 관심 있는 독자는 문서를 직접 참고해 깊이 있게 연구하기 바란다: https://mlir.llvm.org/docs/OpDefinitions/.

## 14. 디버깅 방법
`mlir-tblgen`을 사용해 생성된 텍스트를 살펴본다. TableGen 문법은 때때로 난해할 수 있다. 생성된 텍스트를 읽는 것은 문제를 이해하고 디버깅하는 데 매우 유용하다. `mlir-tblgen`을 빌드하려면 빌드 디렉터리에서 `cmake --build . --target mlir-tblgen`을 실행하고, `bin/` 하위 디렉터리에서 `mlir-tblgen` 바이너리를 찾을 수 있다. 지원되는 모든 generator는 `mlir-tblgen --help`로 확인할 수 있다.

생성된 코드를 보려면 `-I`로 include 경로를 제공하고, 특정 generator를 지정해 `mlir-tblgen`을 호출한다. 예를 들면 다음과 같다.

```cpp
# To see op C++ class declaration
mlir-tblgen --gen-op-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op C++ class definition
mlir-tblgen --gen-op-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op documentation
mlir-tblgen --gen-dialect-doc -I /path/to/mlir/include /path/to/input/td/file

# To see op interface C++ class declaration
mlir-tblgen --gen-op-interface-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op interface C++ class definition
mlir-tblgen --gen-op-interface-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op interface documentation
mlir-tblgen --gen-op-interface-doc -I /path/to/mlir/include /path/to/input/td/file
```



## 15. 정리
이 절에서는 [[밑바닥부터 배우는 딥러닝 컴파일러] 16, MLIR ODS 핵심 정리 상편](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 을 바탕으로 ODS의 핵심 사항들을 보충하여 완성했다. constraint와 attribute의 정의는 모두 MLIR에서 매우 중요한 요소이며, 타입 정의는 개인적으로는 알고만 있어도 충분하다고 본다. 사용자 정의 타입이 필요한 시점에 자세히 연구하면 된다. 마지막으로 MLIR 문법은 다소 난해한데, 입문자는 `mlir-tblgen`을 활용해 디버깅에 도움을 받을 수 있다.

이 두 편의 글에서 나는 MLIR ODS 사양을 한 번 완주하면서 14개의 핵심 사항을 정리했고, 각 핵심 사항마다 OneFlow MLIR의 Op 정의에서 대조해보고 예제 코드와 위치를 함께 제시했다. 독자들의 MLIR 입문에 도움이 되기를 바란다.

