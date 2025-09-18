# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Generic, Protocol, TypeVar, runtime_checkable

from max.interfaces import PipelineInputs

BatchConstructorType = TypeVar(
    "BatchConstructorType", bound=PipelineInputs, covariant=True
)
"""Type variable for batch constructor input types.

This covariant TypeVar is bound to PipelineInputs, ensuring that any type used 
with this variable must be a subtype of PipelineInputs. The covariant nature 
allows for safe substitution of more specific batch types in contexts where
more general types are expected.

Bounds:
    PipelineInputs: All types must be subtypes of the PipelineInputs base class

Variance:
    Covariant: Allows BatchConstructor[SpecificInputs] to be used where 
    BatchConstructor[PipelineInputs] is expected, provided SpecificInputs 
    is a subtype of PipelineInputs.
"""


@runtime_checkable
class BatchConstructor(Protocol, Generic[BatchConstructorType]):
    """Protocol for constructing batches of pipeline inputs.

    This protocol defines the interface for classes that are responsible for
    constructing batches of pipeline inputs from individual input elements.
    The batching strategy is implementation-specific and can vary based on
    the type of pipeline and the nature of the inputs.

    Type Parameters:
        BatchConstructorType: The type of batch inputs produced by this constructor.
            Must be a subtype of PipelineInputs. The type variable is covariant,
            allowing for safe substitution of more specific batch types.

    Example:
        >>> class MyBatchConstructor:
        ...     def construct_batch(self) -> MySpecificInputs:
        ...         # Implementation here
        ...         return batched_inputs
        >>>
        >>> constructor: BatchConstructor[MySpecificInputs] = MyBatchConstructor()
    """

    def construct_batch(self) -> BatchConstructorType:
        """Construct a batch of pipeline inputs for processing.

        This method must be implemented by classes conforming to this protocol
        to define how individual inputs are batched together for efficient pipeline
        processing. The specific batching strategy and input format depend on the
        concrete implementation and the type of pipeline being used.

        Returns:
            BatchConstructorType: A batch of inputs formatted for pipeline processing.
                The exact structure and content depend on the specific pipeline
                input type and batching strategy implemented by the conforming class.
                The returned type must be a subtype of PipelineInputs.

        Note:
            This is a protocol method that must be implemented by any class
            that claims to conform to the BatchConstructor protocol.
        """
        ...
