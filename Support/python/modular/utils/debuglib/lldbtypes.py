# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from typing import Any, Generator

Incomplete = Any


class _SwigNonDynamicMeta(type):
    __setattr__: Incomplete


swig_version: Incomplete


def lldb_iter(obj, getsize, getelem) -> Generator[Incomplete, None, None]:
    ...


INT32_MAX: Incomplete
UINT32_MAX: Incomplete
UINT64_MAX: Incomplete
LLDB_GENERIC_ERROR: Incomplete
LLDB_INVALID_BREAK_ID: Incomplete
LLDB_DEFAULT_BREAK_SIZE: Incomplete
LLDB_INVALID_WATCH_ID: Incomplete
LLDB_WATCH_TYPE_READ: Incomplete
LLDB_WATCH_TYPE_WRITE: Incomplete
LLDB_WATCH_TYPE_MODIFY: Incomplete
LLDB_INVALID_SITE_ID: Incomplete
LLDB_REGNUM_GENERIC_PC: Incomplete
LLDB_REGNUM_GENERIC_SP: Incomplete
LLDB_REGNUM_GENERIC_FP: Incomplete
LLDB_REGNUM_GENERIC_RA: Incomplete
LLDB_REGNUM_GENERIC_FLAGS: Incomplete
LLDB_REGNUM_GENERIC_ARG1: Incomplete
LLDB_REGNUM_GENERIC_ARG2: Incomplete
LLDB_REGNUM_GENERIC_ARG3: Incomplete
LLDB_REGNUM_GENERIC_ARG4: Incomplete
LLDB_REGNUM_GENERIC_ARG5: Incomplete
LLDB_REGNUM_GENERIC_ARG6: Incomplete
LLDB_REGNUM_GENERIC_ARG7: Incomplete
LLDB_REGNUM_GENERIC_ARG8: Incomplete
LLDB_REGNUM_GENERIC_TP: Incomplete
LLDB_INVALID_STOP_ID: Incomplete
LLDB_INVALID_ADDRESS: Incomplete
LLDB_INVALID_INDEX32: Incomplete
LLDB_INVALID_IVAR_OFFSET: Incomplete
LLDB_INVALID_IMAGE_TOKEN: Incomplete
LLDB_INVALID_MODULE_VERSION: Incomplete
LLDB_INVALID_REGNUM: Incomplete
LLDB_INVALID_UID: Incomplete
LLDB_INVALID_PROCESS_ID: Incomplete
LLDB_INVALID_THREAD_ID: Incomplete
LLDB_INVALID_FRAME_ID: Incomplete
LLDB_INVALID_SIGNAL_NUMBER: Incomplete
LLDB_INVALID_OFFSET: Incomplete
LLDB_INVALID_LINE_NUMBER: Incomplete
LLDB_INVALID_COLUMN_NUMBER: Incomplete
LLDB_INVALID_QUEUE_ID: Incomplete
LLDB_INVALID_CPU_ID: Incomplete
LLDB_INVALID_WATCHPOINT_RESOURCE_ID: Incomplete
LLDB_ARCH_DEFAULT: Incomplete
LLDB_ARCH_DEFAULT_32BIT: Incomplete
LLDB_ARCH_DEFAULT_64BIT: Incomplete
LLDB_INVALID_CPUTYPE: Incomplete
LLDB_MAX_NUM_OPTION_SETS: Incomplete
LLDB_OPT_SET_ALL: Incomplete
LLDB_OPT_SET_1: Incomplete
LLDB_OPT_SET_2: Incomplete
LLDB_OPT_SET_3: Incomplete
LLDB_OPT_SET_4: Incomplete
LLDB_OPT_SET_5: Incomplete
LLDB_OPT_SET_6: Incomplete
LLDB_OPT_SET_7: Incomplete
LLDB_OPT_SET_8: Incomplete
LLDB_OPT_SET_9: Incomplete
LLDB_OPT_SET_10: Incomplete
LLDB_OPT_SET_11: Incomplete
LLDB_OPT_SET_12: Incomplete
eStateInvalid: Incomplete
eStateUnloaded: Incomplete
eStateConnected: Incomplete
eStateAttaching: Incomplete
eStateLaunching: Incomplete
eStateStopped: Incomplete
eStateRunning: Incomplete
eStateStepping: Incomplete
eStateCrashed: Incomplete
eStateDetached: Incomplete
eStateExited: Incomplete
eStateSuspended: Incomplete
kLastStateType: Incomplete
eLaunchFlagNone: Incomplete
eLaunchFlagExec: Incomplete
eLaunchFlagDebug: Incomplete
eLaunchFlagStopAtEntry: Incomplete
eLaunchFlagDisableASLR: Incomplete
eLaunchFlagDisableSTDIO: Incomplete
eLaunchFlagLaunchInTTY: Incomplete
eLaunchFlagLaunchInShell: Incomplete
eLaunchFlagLaunchInSeparateProcessGroup: Incomplete
eLaunchFlagDontSetExitStatus: Incomplete
eLaunchFlagDetachOnError: Incomplete
eLaunchFlagShellExpandArguments: Incomplete
eLaunchFlagCloseTTYOnExit: Incomplete
eLaunchFlagInheritTCCFromParent: Incomplete
eOnlyThisThread: Incomplete
eAllThreads: Incomplete
eOnlyDuringStepping: Incomplete
eByteOrderInvalid: Incomplete
eByteOrderBig: Incomplete
eByteOrderPDP: Incomplete
eByteOrderLittle: Incomplete
eEncodingInvalid: Incomplete
eEncodingUint: Incomplete
eEncodingSint: Incomplete
eEncodingIEEE754: Incomplete
eEncodingVector: Incomplete
eFormatDefault: Incomplete
eFormatInvalid: Incomplete
eFormatBoolean: Incomplete
eFormatBinary: Incomplete
eFormatBytes: Incomplete
eFormatBytesWithASCII: Incomplete
eFormatChar: Incomplete
eFormatCharPrintable: Incomplete
eFormatComplex: Incomplete
eFormatComplexFloat: Incomplete
eFormatCString: Incomplete
eFormatDecimal: Incomplete
eFormatEnum: Incomplete
eFormatHex: Incomplete
eFormatHexUppercase: Incomplete
eFormatFloat: Incomplete
eFormatOctal: Incomplete
eFormatOSType: Incomplete
eFormatUnicode16: Incomplete
eFormatUnicode32: Incomplete
eFormatUnsigned: Incomplete
eFormatPointer: Incomplete
eFormatVectorOfChar: Incomplete
eFormatVectorOfSInt8: Incomplete
eFormatVectorOfUInt8: Incomplete
eFormatVectorOfSInt16: Incomplete
eFormatVectorOfUInt16: Incomplete
eFormatVectorOfSInt32: Incomplete
eFormatVectorOfUInt32: Incomplete
eFormatVectorOfSInt64: Incomplete
eFormatVectorOfUInt64: Incomplete
eFormatVectorOfFloat16: Incomplete
eFormatVectorOfFloat32: Incomplete
eFormatVectorOfFloat64: Incomplete
eFormatVectorOfUInt128: Incomplete
eFormatComplexInteger: Incomplete
eFormatCharArray: Incomplete
eFormatAddressInfo: Incomplete
eFormatHexFloat: Incomplete
eFormatInstruction: Incomplete
eFormatVoid: Incomplete
eFormatUnicode8: Incomplete
kNumFormats: Incomplete
eDescriptionLevelBrief: Incomplete
eDescriptionLevelFull: Incomplete
eDescriptionLevelVerbose: Incomplete
eDescriptionLevelInitial: Incomplete
kNumDescriptionLevels: Incomplete
eScriptLanguageNone: Incomplete
eScriptLanguagePython: Incomplete
eScriptLanguageLua: Incomplete
eScriptLanguageUnknown: Incomplete
eScriptLanguageDefault: Incomplete
eRegisterKindEHFrame: Incomplete
eRegisterKindDWARF: Incomplete
eRegisterKindGeneric: Incomplete
eRegisterKindProcessPlugin: Incomplete
eRegisterKindLLDB: Incomplete
kNumRegisterKinds: Incomplete
eStopReasonInvalid: Incomplete
eStopReasonNone: Incomplete
eStopReasonTrace: Incomplete
eStopReasonBreakpoint: Incomplete
eStopReasonWatchpoint: Incomplete
eStopReasonSignal: Incomplete
eStopReasonException: Incomplete
eStopReasonExec: Incomplete
eStopReasonPlanComplete: Incomplete
eStopReasonThreadExiting: Incomplete
eStopReasonInstrumentation: Incomplete
eStopReasonProcessorTrace: Incomplete
eStopReasonFork: Incomplete
eStopReasonVFork: Incomplete
eStopReasonVForkDone: Incomplete
eReturnStatusInvalid: Incomplete
eReturnStatusSuccessFinishNoResult: Incomplete
eReturnStatusSuccessFinishResult: Incomplete
eReturnStatusSuccessContinuingNoResult: Incomplete
eReturnStatusSuccessContinuingResult: Incomplete
eReturnStatusStarted: Incomplete
eReturnStatusFailed: Incomplete
eReturnStatusQuit: Incomplete
eExpressionCompleted: Incomplete
eExpressionSetupError: Incomplete
eExpressionParseError: Incomplete
eExpressionDiscarded: Incomplete
eExpressionInterrupted: Incomplete
eExpressionHitBreakpoint: Incomplete
eExpressionTimedOut: Incomplete
eExpressionResultUnavailable: Incomplete
eExpressionStoppedForDebug: Incomplete
eExpressionThreadVanished: Incomplete
eSearchDepthInvalid: Incomplete
eSearchDepthTarget: Incomplete
eSearchDepthModule: Incomplete
eSearchDepthCompUnit: Incomplete
eSearchDepthFunction: Incomplete
eSearchDepthBlock: Incomplete
eSearchDepthAddress: Incomplete
kLastSearchDepthKind: Incomplete
eConnectionStatusSuccess: Incomplete
eConnectionStatusEndOfFile: Incomplete
eConnectionStatusError: Incomplete
eConnectionStatusTimedOut: Incomplete
eConnectionStatusNoConnection: Incomplete
eConnectionStatusLostConnection: Incomplete
eConnectionStatusInterrupted: Incomplete
eErrorTypeInvalid: Incomplete
eErrorTypeGeneric: Incomplete
eErrorTypeMachKernel: Incomplete
eErrorTypePOSIX: Incomplete
eErrorTypeExpression: Incomplete
eErrorTypeWin32: Incomplete
eValueTypeInvalid: Incomplete
eValueTypeVariableGlobal: Incomplete
eValueTypeVariableStatic: Incomplete
eValueTypeVariableArgument: Incomplete
eValueTypeVariableLocal: Incomplete
eValueTypeRegister: Incomplete
eValueTypeRegisterSet: Incomplete
eValueTypeConstResult: Incomplete
eValueTypeVariableThreadLocal: Incomplete
eValueTypeVTable: Incomplete
eValueTypeVTableEntry: Incomplete
eInputReaderGranularityInvalid: Incomplete
eInputReaderGranularityByte: Incomplete
eInputReaderGranularityWord: Incomplete
eInputReaderGranularityLine: Incomplete
eInputReaderGranularityAll: Incomplete
eSymbolContextTarget: Incomplete
eSymbolContextModule: Incomplete
eSymbolContextCompUnit: Incomplete
eSymbolContextFunction: Incomplete
eSymbolContextBlock: Incomplete
eSymbolContextLineEntry: Incomplete
eSymbolContextSymbol: Incomplete
eSymbolContextEverything: Incomplete
eSymbolContextVariable: Incomplete
eSymbolContextLastItem: Incomplete
ePermissionsWritable: Incomplete
ePermissionsReadable: Incomplete
ePermissionsExecutable: Incomplete
eInputReaderActivate: Incomplete
eInputReaderAsynchronousOutputWritten: Incomplete
eInputReaderReactivate: Incomplete
eInputReaderDeactivate: Incomplete
eInputReaderGotToken: Incomplete
eInputReaderInterrupt: Incomplete
eInputReaderEndOfFile: Incomplete
eInputReaderDone: Incomplete
eBreakpointEventTypeInvalidType: Incomplete
eBreakpointEventTypeAdded: Incomplete
eBreakpointEventTypeRemoved: Incomplete
eBreakpointEventTypeLocationsAdded: Incomplete
eBreakpointEventTypeLocationsRemoved: Incomplete
eBreakpointEventTypeLocationsResolved: Incomplete
eBreakpointEventTypeEnabled: Incomplete
eBreakpointEventTypeDisabled: Incomplete
eBreakpointEventTypeCommandChanged: Incomplete
eBreakpointEventTypeConditionChanged: Incomplete
eBreakpointEventTypeIgnoreChanged: Incomplete
eBreakpointEventTypeThreadChanged: Incomplete
eBreakpointEventTypeAutoContinueChanged: Incomplete
eWatchpointEventTypeInvalidType: Incomplete
eWatchpointEventTypeAdded: Incomplete
eWatchpointEventTypeRemoved: Incomplete
eWatchpointEventTypeEnabled: Incomplete
eWatchpointEventTypeDisabled: Incomplete
eWatchpointEventTypeCommandChanged: Incomplete
eWatchpointEventTypeConditionChanged: Incomplete
eWatchpointEventTypeIgnoreChanged: Incomplete
eWatchpointEventTypeThreadChanged: Incomplete
eWatchpointEventTypeTypeChanged: Incomplete
eWatchpointWriteTypeDisabled: Incomplete
eWatchpointWriteTypeAlways: Incomplete
eWatchpointWriteTypeOnModify: Incomplete
eLanguageTypeUnknown: Incomplete
eLanguageTypeC89: Incomplete
eLanguageTypeC: Incomplete
eLanguageTypeAda83: Incomplete
eLanguageTypeC_plus_plus: Incomplete
eLanguageTypeCobol74: Incomplete
eLanguageTypeCobol85: Incomplete
eLanguageTypeFortran77: Incomplete
eLanguageTypeFortran90: Incomplete
eLanguageTypePascal83: Incomplete
eLanguageTypeModula2: Incomplete
eLanguageTypeJava: Incomplete
eLanguageTypeC99: Incomplete
eLanguageTypeAda95: Incomplete
eLanguageTypeFortran95: Incomplete
eLanguageTypePLI: Incomplete
eLanguageTypeObjC: Incomplete
eLanguageTypeObjC_plus_plus: Incomplete
eLanguageTypeUPC: Incomplete
eLanguageTypeD: Incomplete
eLanguageTypePython: Incomplete
eLanguageTypeOpenCL: Incomplete
eLanguageTypeGo: Incomplete
eLanguageTypeModula3: Incomplete
eLanguageTypeHaskell: Incomplete
eLanguageTypeC_plus_plus_03: Incomplete
eLanguageTypeC_plus_plus_11: Incomplete
eLanguageTypeOCaml: Incomplete
eLanguageTypeRust: Incomplete
eLanguageTypeC11: Incomplete
eLanguageTypeSwift: Incomplete
eLanguageTypeJulia: Incomplete
eLanguageTypeDylan: Incomplete
eLanguageTypeC_plus_plus_14: Incomplete
eLanguageTypeFortran03: Incomplete
eLanguageTypeFortran08: Incomplete
eLanguageTypeRenderScript: Incomplete
eLanguageTypeBLISS: Incomplete
eLanguageTypeKotlin: Incomplete
eLanguageTypeZig: Incomplete
eLanguageTypeCrystal: Incomplete
eLanguageTypeC_plus_plus_17: Incomplete
eLanguageTypeC_plus_plus_20: Incomplete
eLanguageTypeC17: Incomplete
eLanguageTypeFortran18: Incomplete
eLanguageTypeAda2005: Incomplete
eLanguageTypeAda2012: Incomplete
eLanguageTypeHIP: Incomplete
eLanguageTypeAssembly: Incomplete
eLanguageTypeC_sharp: Incomplete
eLanguageTypeMojo: Incomplete
eLanguageTypeMipsAssembler: Incomplete
eNumLanguageTypes: Incomplete
eInstrumentationRuntimeTypeAddressSanitizer: Incomplete
eInstrumentationRuntimeTypeThreadSanitizer: Incomplete
eInstrumentationRuntimeTypeUndefinedBehaviorSanitizer: Incomplete
eInstrumentationRuntimeTypeMainThreadChecker: Incomplete
eInstrumentationRuntimeTypeSwiftRuntimeReporting: Incomplete
eInstrumentationRuntimeTypeLibsanitizersAsan: Incomplete
eNumInstrumentationRuntimeTypes: Incomplete
eNoDynamicValues: Incomplete
eDynamicCanRunTarget: Incomplete
eDynamicDontRunTarget: Incomplete
eStopShowColumnAnsiOrCaret: Incomplete
eStopShowColumnAnsi: Incomplete
eStopShowColumnCaret: Incomplete
eStopShowColumnNone: Incomplete
eAccessNone: Incomplete
eAccessPublic: Incomplete
eAccessPrivate: Incomplete
eAccessProtected: Incomplete
eAccessPackage: Incomplete
eArgTypeAddress: Incomplete
eArgTypeAddressOrExpression: Incomplete
eArgTypeAliasName: Incomplete
eArgTypeAliasOptions: Incomplete
eArgTypeArchitecture: Incomplete
eArgTypeBoolean: Incomplete
eArgTypeBreakpointID: Incomplete
eArgTypeBreakpointIDRange: Incomplete
eArgTypeBreakpointName: Incomplete
eArgTypeByteSize: Incomplete
eArgTypeClassName: Incomplete
eArgTypeCommandName: Incomplete
eArgTypeCount: Incomplete
eArgTypeDescriptionVerbosity: Incomplete
eArgTypeDirectoryName: Incomplete
eArgTypeDisassemblyFlavor: Incomplete
eArgTypeEndAddress: Incomplete
eArgTypeExpression: Incomplete
eArgTypeExpressionPath: Incomplete
eArgTypeExprFormat: Incomplete
eArgTypeFileLineColumn: Incomplete
eArgTypeFilename: Incomplete
eArgTypeFormat: Incomplete
eArgTypeFrameIndex: Incomplete
eArgTypeFullName: Incomplete
eArgTypeFunctionName: Incomplete
eArgTypeFunctionOrSymbol: Incomplete
eArgTypeGDBFormat: Incomplete
eArgTypeHelpText: Incomplete
eArgTypeIndex: Incomplete
eArgTypeLanguage: Incomplete
eArgTypeLineNum: Incomplete
eArgTypeLogCategory: Incomplete
eArgTypeLogChannel: Incomplete
eArgTypeMethod: Incomplete
eArgTypeName: Incomplete
eArgTypeNewPathPrefix: Incomplete
eArgTypeNumLines: Incomplete
eArgTypeNumberPerLine: Incomplete
eArgTypeOffset: Incomplete
eArgTypeOldPathPrefix: Incomplete
eArgTypeOneLiner: Incomplete
eArgTypePath: Incomplete
eArgTypePermissionsNumber: Incomplete
eArgTypePermissionsString: Incomplete
eArgTypePid: Incomplete
eArgTypePlugin: Incomplete
eArgTypeProcessName: Incomplete
eArgTypePythonClass: Incomplete
eArgTypePythonFunction: Incomplete
eArgTypePythonScript: Incomplete
eArgTypeQueueName: Incomplete
eArgTypeRegisterName: Incomplete
eArgTypeRegularExpression: Incomplete
eArgTypeRunArgs: Incomplete
eArgTypeRunMode: Incomplete
eArgTypeScriptedCommandSynchronicity: Incomplete
eArgTypeScriptLang: Incomplete
eArgTypeSearchWord: Incomplete
eArgTypeSelector: Incomplete
eArgTypeSettingIndex: Incomplete
eArgTypeSettingKey: Incomplete
eArgTypeSettingPrefix: Incomplete
eArgTypeSettingVariableName: Incomplete
eArgTypeShlibName: Incomplete
eArgTypeSourceFile: Incomplete
eArgTypeSortOrder: Incomplete
eArgTypeStartAddress: Incomplete
eArgTypeSummaryString: Incomplete
eArgTypeSymbol: Incomplete
eArgTypeThreadID: Incomplete
eArgTypeThreadIndex: Incomplete
eArgTypeThreadName: Incomplete
eArgTypeTypeName: Incomplete
eArgTypeUnsignedInteger: Incomplete
eArgTypeUnixSignal: Incomplete
eArgTypeVarName: Incomplete
eArgTypeValue: Incomplete
eArgTypeWidth: Incomplete
eArgTypeNone: Incomplete
eArgTypePlatform: Incomplete
eArgTypeWatchpointID: Incomplete
eArgTypeWatchpointIDRange: Incomplete
eArgTypeWatchType: Incomplete
eArgRawInput: Incomplete
eArgTypeCommand: Incomplete
eArgTypeColumnNum: Incomplete
eArgTypeModuleUUID: Incomplete
eArgTypeSaveCoreStyle: Incomplete
eArgTypeLogHandler: Incomplete
eArgTypeSEDStylePair: Incomplete
eArgTypeRecognizerID: Incomplete
eArgTypeConnectURL: Incomplete
eArgTypeTargetID: Incomplete
eArgTypeStopHookID: Incomplete
eArgTypeCompletionType: Incomplete
eArgTypeLastArg: Incomplete
eSymbolTypeAny: Incomplete
eSymbolTypeInvalid: Incomplete
eSymbolTypeAbsolute: Incomplete
eSymbolTypeCode: Incomplete
eSymbolTypeResolver: Incomplete
eSymbolTypeData: Incomplete
eSymbolTypeTrampoline: Incomplete
eSymbolTypeRuntime: Incomplete
eSymbolTypeException: Incomplete
eSymbolTypeSourceFile: Incomplete
eSymbolTypeHeaderFile: Incomplete
eSymbolTypeObjectFile: Incomplete
eSymbolTypeCommonBlock: Incomplete
eSymbolTypeBlock: Incomplete
eSymbolTypeLocal: Incomplete
eSymbolTypeParam: Incomplete
eSymbolTypeVariable: Incomplete
eSymbolTypeVariableType: Incomplete
eSymbolTypeLineEntry: Incomplete
eSymbolTypeLineHeader: Incomplete
eSymbolTypeScopeBegin: Incomplete
eSymbolTypeScopeEnd: Incomplete
eSymbolTypeAdditional: Incomplete
eSymbolTypeCompiler: Incomplete
eSymbolTypeInstrumentation: Incomplete
eSymbolTypeUndefined: Incomplete
eSymbolTypeObjCClass: Incomplete
eSymbolTypeObjCMetaClass: Incomplete
eSymbolTypeObjCIVar: Incomplete
eSymbolTypeReExported: Incomplete
eSectionTypeInvalid: Incomplete
eSectionTypeCode: Incomplete
eSectionTypeContainer: Incomplete
eSectionTypeData: Incomplete
eSectionTypeDataCString: Incomplete
eSectionTypeDataCStringPointers: Incomplete
eSectionTypeDataSymbolAddress: Incomplete
eSectionTypeData4: Incomplete
eSectionTypeData8: Incomplete
eSectionTypeData16: Incomplete
eSectionTypeDataPointers: Incomplete
eSectionTypeDebug: Incomplete
eSectionTypeZeroFill: Incomplete
eSectionTypeDataObjCMessageRefs: Incomplete
eSectionTypeDataObjCCFStrings: Incomplete
eSectionTypeDWARFDebugAbbrev: Incomplete
eSectionTypeDWARFDebugAddr: Incomplete
eSectionTypeDWARFDebugAranges: Incomplete
eSectionTypeDWARFDebugCuIndex: Incomplete
eSectionTypeDWARFDebugFrame: Incomplete
eSectionTypeDWARFDebugInfo: Incomplete
eSectionTypeDWARFDebugLine: Incomplete
eSectionTypeDWARFDebugLoc: Incomplete
eSectionTypeDWARFDebugMacInfo: Incomplete
eSectionTypeDWARFDebugMacro: Incomplete
eSectionTypeDWARFDebugPubNames: Incomplete
eSectionTypeDWARFDebugPubTypes: Incomplete
eSectionTypeDWARFDebugRanges: Incomplete
eSectionTypeDWARFDebugStr: Incomplete
eSectionTypeDWARFDebugStrOffsets: Incomplete
eSectionTypeDWARFAppleNames: Incomplete
eSectionTypeDWARFAppleTypes: Incomplete
eSectionTypeDWARFAppleNamespaces: Incomplete
eSectionTypeDWARFAppleObjC: Incomplete
eSectionTypeELFSymbolTable: Incomplete
eSectionTypeELFDynamicSymbols: Incomplete
eSectionTypeELFRelocationEntries: Incomplete
eSectionTypeELFDynamicLinkInfo: Incomplete
eSectionTypeEHFrame: Incomplete
eSectionTypeARMexidx: Incomplete
eSectionTypeARMextab: Incomplete
eSectionTypeCompactUnwind: Incomplete
eSectionTypeGoSymtab: Incomplete
eSectionTypeAbsoluteAddress: Incomplete
eSectionTypeDWARFGNUDebugAltLink: Incomplete
eSectionTypeDWARFDebugTypes: Incomplete
eSectionTypeDWARFDebugNames: Incomplete
eSectionTypeOther: Incomplete
eSectionTypeDWARFDebugLineStr: Incomplete
eSectionTypeDWARFDebugRngLists: Incomplete
eSectionTypeDWARFDebugLocLists: Incomplete
eSectionTypeDWARFDebugAbbrevDwo: Incomplete
eSectionTypeDWARFDebugInfoDwo: Incomplete
eSectionTypeDWARFDebugStrDwo: Incomplete
eSectionTypeDWARFDebugStrOffsetsDwo: Incomplete
eSectionTypeDWARFDebugTypesDwo: Incomplete
eSectionTypeDWARFDebugRngListsDwo: Incomplete
eSectionTypeDWARFDebugLocDwo: Incomplete
eSectionTypeDWARFDebugLocListsDwo: Incomplete
eSectionTypeDWARFDebugTuIndex: Incomplete
eSectionTypeCTF: Incomplete
eSectionTypeSwiftModules: Incomplete
eEmulateInstructionOptionNone: Incomplete
eEmulateInstructionOptionAutoAdvancePC: Incomplete
eEmulateInstructionOptionIgnoreConditions: Incomplete
eFunctionNameTypeNone: Incomplete
eFunctionNameTypeAuto: Incomplete
eFunctionNameTypeFull: Incomplete
eFunctionNameTypeBase: Incomplete
eFunctionNameTypeMethod: Incomplete
eFunctionNameTypeSelector: Incomplete
eFunctionNameTypeAny: Incomplete
eBasicTypeInvalid: Incomplete
eBasicTypeVoid: Incomplete
eBasicTypeChar: Incomplete
eBasicTypeSignedChar: Incomplete
eBasicTypeUnsignedChar: Incomplete
eBasicTypeWChar: Incomplete
eBasicTypeSignedWChar: Incomplete
eBasicTypeUnsignedWChar: Incomplete
eBasicTypeChar16: Incomplete
eBasicTypeChar32: Incomplete
eBasicTypeChar8: Incomplete
eBasicTypeShort: Incomplete
eBasicTypeUnsignedShort: Incomplete
eBasicTypeInt: Incomplete
eBasicTypeUnsignedInt: Incomplete
eBasicTypeLong: Incomplete
eBasicTypeUnsignedLong: Incomplete
eBasicTypeLongLong: Incomplete
eBasicTypeUnsignedLongLong: Incomplete
eBasicTypeInt128: Incomplete
eBasicTypeUnsignedInt128: Incomplete
eBasicTypeBool: Incomplete
eBasicTypeHalf: Incomplete
eBasicTypeFloat: Incomplete
eBasicTypeDouble: Incomplete
eBasicTypeLongDouble: Incomplete
eBasicTypeFloatComplex: Incomplete
eBasicTypeDoubleComplex: Incomplete
eBasicTypeLongDoubleComplex: Incomplete
eBasicTypeObjCID: Incomplete
eBasicTypeObjCClass: Incomplete
eBasicTypeObjCSel: Incomplete
eBasicTypeNullPtr: Incomplete
eBasicTypeOther: Incomplete
eTraceTypeNone: Incomplete
eTraceTypeProcessorTrace: Incomplete
eStructuredDataTypeInvalid: Incomplete
eStructuredDataTypeNull: Incomplete
eStructuredDataTypeGeneric: Incomplete
eStructuredDataTypeArray: Incomplete
eStructuredDataTypeInteger: Incomplete
eStructuredDataTypeFloat: Incomplete
eStructuredDataTypeBoolean: Incomplete
eStructuredDataTypeString: Incomplete
eStructuredDataTypeDictionary: Incomplete
eStructuredDataTypeSignedInteger: Incomplete
eStructuredDataTypeUnsignedInteger: Incomplete
eTypeClassInvalid: Incomplete
eTypeClassArray: Incomplete
eTypeClassBlockPointer: Incomplete
eTypeClassBuiltin: Incomplete
eTypeClassClass: Incomplete
eTypeClassComplexFloat: Incomplete
eTypeClassComplexInteger: Incomplete
eTypeClassEnumeration: Incomplete
eTypeClassFunction: Incomplete
eTypeClassMemberPointer: Incomplete
eTypeClassObjCObject: Incomplete
eTypeClassObjCInterface: Incomplete
eTypeClassObjCObjectPointer: Incomplete
eTypeClassPointer: Incomplete
eTypeClassReference: Incomplete
eTypeClassStruct: Incomplete
eTypeClassTypedef: Incomplete
eTypeClassUnion: Incomplete
eTypeClassVector: Incomplete
eTypeClassOther: Incomplete
eTypeClassAny: Incomplete
eTemplateArgumentKindNull: Incomplete
eTemplateArgumentKindType: Incomplete
eTemplateArgumentKindDeclaration: Incomplete
eTemplateArgumentKindIntegral: Incomplete
eTemplateArgumentKindTemplate: Incomplete
eTemplateArgumentKindTemplateExpansion: Incomplete
eTemplateArgumentKindExpression: Incomplete
eTemplateArgumentKindPack: Incomplete
eTemplateArgumentKindNullPtr: Incomplete
eFormatterMatchExact: Incomplete
eFormatterMatchRegex: Incomplete
eFormatterMatchCallback: Incomplete
eLastFormatterMatchType: Incomplete
eTypeOptionNone: Incomplete
eTypeOptionCascade: Incomplete
eTypeOptionSkipPointers: Incomplete
eTypeOptionSkipReferences: Incomplete
eTypeOptionHideChildren: Incomplete
eTypeOptionHideValue: Incomplete
eTypeOptionShowOneLiner: Incomplete
eTypeOptionHideNames: Incomplete
eTypeOptionNonCacheable: Incomplete
eTypeOptionHideEmptyAggregates: Incomplete
eTypeOptionFrontEndWantsDereference: Incomplete
eFrameCompareInvalid: Incomplete
eFrameCompareUnknown: Incomplete
eFrameCompareEqual: Incomplete
eFrameCompareSameParent: Incomplete
eFrameCompareYounger: Incomplete
eFrameCompareOlder: Incomplete
eFilePermissionsUserRead: Incomplete
eFilePermissionsUserWrite: Incomplete
eFilePermissionsUserExecute: Incomplete
eFilePermissionsGroupRead: Incomplete
eFilePermissionsGroupWrite: Incomplete
eFilePermissionsGroupExecute: Incomplete
eFilePermissionsWorldRead: Incomplete
eFilePermissionsWorldWrite: Incomplete
eFilePermissionsWorldExecute: Incomplete
eFilePermissionsUserRW: Incomplete
eFileFilePermissionsUserRX: Incomplete
eFilePermissionsUserRWX: Incomplete
eFilePermissionsGroupRW: Incomplete
eFilePermissionsGroupRX: Incomplete
eFilePermissionsGroupRWX: Incomplete
eFilePermissionsWorldRW: Incomplete
eFilePermissionsWorldRX: Incomplete
eFilePermissionsWorldRWX: Incomplete
eFilePermissionsEveryoneR: Incomplete
eFilePermissionsEveryoneW: Incomplete
eFilePermissionsEveryoneX: Incomplete
eFilePermissionsEveryoneRW: Incomplete
eFilePermissionsEveryoneRX: Incomplete
eFilePermissionsEveryoneRWX: Incomplete
eFilePermissionsFileDefault: Incomplete
eFilePermissionsDirectoryDefault: Incomplete
eQueueItemKindUnknown: Incomplete
eQueueItemKindFunction: Incomplete
eQueueItemKindBlock: Incomplete
eQueueKindUnknown: Incomplete
eQueueKindSerial: Incomplete
eQueueKindConcurrent: Incomplete
eExpressionEvaluationParse: Incomplete
eExpressionEvaluationIRGen: Incomplete
eExpressionEvaluationExecution: Incomplete
eExpressionEvaluationComplete: Incomplete
eInstructionControlFlowKindUnknown: Incomplete
eInstructionControlFlowKindOther: Incomplete
eInstructionControlFlowKindCall: Incomplete
eInstructionControlFlowKindReturn: Incomplete
eInstructionControlFlowKindJump: Incomplete
eInstructionControlFlowKindCondJump: Incomplete
eInstructionControlFlowKindFarCall: Incomplete
eInstructionControlFlowKindFarReturn: Incomplete
eInstructionControlFlowKindFarJump: Incomplete
eWatchpointKindWrite: Incomplete
eWatchpointKindRead: Incomplete
eGdbSignalBadAccess: Incomplete
eGdbSignalBadInstruction: Incomplete
eGdbSignalArithmetic: Incomplete
eGdbSignalEmulation: Incomplete
eGdbSignalSoftware: Incomplete
eGdbSignalBreakpoint: Incomplete
ePathTypeLLDBShlibDir: Incomplete
ePathTypeSupportExecutableDir: Incomplete
ePathTypeHeaderDir: Incomplete
ePathTypePythonDir: Incomplete
ePathTypeLLDBSystemPlugins: Incomplete
ePathTypeLLDBUserPlugins: Incomplete
ePathTypeLLDBTempSystemDir: Incomplete
ePathTypeGlobalLLDBTempSystemDir: Incomplete
ePathTypeClangDir: Incomplete
eMemberFunctionKindUnknown: Incomplete
eMemberFunctionKindConstructor: Incomplete
eMemberFunctionKindDestructor: Incomplete
eMemberFunctionKindInstanceMethod: Incomplete
eMemberFunctionKindStaticMethod: Incomplete
eMatchTypeNormal: Incomplete
eMatchTypeRegex: Incomplete
eMatchTypeStartsWith: Incomplete
eTypeHasChildren: Incomplete
eTypeHasValue: Incomplete
eTypeIsArray: Incomplete
eTypeIsBlock: Incomplete
eTypeIsBuiltIn: Incomplete
eTypeIsClass: Incomplete
eTypeIsCPlusPlus: Incomplete
eTypeIsEnumeration: Incomplete
eTypeIsFuncPrototype: Incomplete
eTypeIsMember: Incomplete
eTypeIsObjC: Incomplete
eTypeIsPointer: Incomplete
eTypeIsReference: Incomplete
eTypeIsStructUnion: Incomplete
eTypeIsTemplate: Incomplete
eTypeIsTypedef: Incomplete
eTypeIsVector: Incomplete
eTypeIsScalar: Incomplete
eTypeIsInteger: Incomplete
eTypeIsFloat: Incomplete
eTypeIsComplex: Incomplete
eTypeIsSigned: Incomplete
eTypeInstanceIsPointer: Incomplete
eCommandRequiresTarget: Incomplete
eCommandRequiresProcess: Incomplete
eCommandRequiresThread: Incomplete
eCommandRequiresFrame: Incomplete
eCommandRequiresRegContext: Incomplete
eCommandTryTargetAPILock: Incomplete
eCommandProcessMustBeLaunched: Incomplete
eCommandProcessMustBePaused: Incomplete
eCommandProcessMustBeTraced: Incomplete
eTypeSummaryCapped: Incomplete
eTypeSummaryUncapped: Incomplete
eCommandInterpreterResultSuccess: Incomplete
eCommandInterpreterResultInferiorCrash: Incomplete
eCommandInterpreterResultCommandError: Incomplete
eCommandInterpreterResultQuitRequested: Incomplete
eSaveCoreUnspecified: Incomplete
eSaveCoreFull: Incomplete
eSaveCoreDirtyOnly: Incomplete
eSaveCoreStackOnly: Incomplete
eTraceEventDisabledSW: Incomplete
eTraceEventDisabledHW: Incomplete
eTraceEventCPUChanged: Incomplete
eTraceEventHWClockTick: Incomplete
eTraceEventSyncPoint: Incomplete
eTraceItemKindError: Incomplete
eTraceItemKindEvent: Incomplete
eTraceItemKindInstruction: Incomplete
eTraceCursorSeekTypeBeginning: Incomplete
eTraceCursorSeekTypeCurrent: Incomplete
eTraceCursorSeekTypeEnd: Incomplete
eDWIMPrintVerbosityNone: Incomplete
eDWIMPrintVerbosityExpression: Incomplete
eDWIMPrintVerbosityFull: Incomplete
eWatchPointValueKindInvalid: Incomplete
eWatchPointValueKindVariable: Incomplete
eWatchPointValueKindExpression: Incomplete
eNoCompletion: Incomplete
eSourceFileCompletion: Incomplete
eDiskFileCompletion: Incomplete
eDiskDirectoryCompletion: Incomplete
eSymbolCompletion: Incomplete
eModuleCompletion: Incomplete
eSettingsNameCompletion: Incomplete
ePlatformPluginCompletion: Incomplete
eArchitectureCompletion: Incomplete
eVariablePathCompletion: Incomplete
eRegisterCompletion: Incomplete
eBreakpointCompletion: Incomplete
eProcessPluginCompletion: Incomplete
eDisassemblyFlavorCompletion: Incomplete
eTypeLanguageCompletion: Incomplete
eFrameIndexCompletion: Incomplete
eModuleUUIDCompletion: Incomplete
eStopHookIDCompletion: Incomplete
eThreadIndexCompletion: Incomplete
eWatchpointIDCompletion: Incomplete
eBreakpointNameCompletion: Incomplete
eProcessIDCompletion: Incomplete
eProcessNameCompletion: Incomplete
eRemoteDiskFileCompletion: Incomplete
eRemoteDiskDirectoryCompletion: Incomplete
eTypeCategoryNameCompletion: Incomplete
eCustomCompletion: Incomplete
eThreadIDCompletion: Incomplete
eTerminatorCompletion: Incomplete


class SBAddress:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetFileAddress(self):
        ...

    def GetLoadAddress(self, target):
        ...

    def SetAddress(self, section, offset):
        ...

    def SetLoadAddress(self, load_addr, target):
        ...

    def OffsetAddress(self, offset):
        ...

    def GetDescription(self, description):
        ...

    def GetSymbolContext(self, resolve_scope):
        ...

    def GetSection(self):
        ...

    def GetOffset(self):
        ...

    def GetModule(self):
        ...

    def GetCompileUnit(self):
        ...

    def GetFunction(self):
        ...

    def GetBlock(self):
        ...

    def GetSymbol(self):
        ...

    def GetLineEntry(self):
        ...

    def __get_load_addr_property__(self):
        ...

    def __set_load_addr_property__(self, load_addr):
        ...

    def __int__(self) -> int:
        ...

    def __oct__(self):
        ...

    def __hex__(self):
        ...

    module: Incomplete
    compile_unit: Incomplete
    line_entry: Incomplete
    function: Incomplete
    block: Incomplete
    symbol: Incomplete
    offset: Incomplete
    section: Incomplete
    file_addr: Incomplete
    load_addr: Incomplete


class SBAttachInfo:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetProcessID(self):
        ...

    def SetProcessID(self, pid):
        ...

    def SetExecutable(self, *args):
        ...

    def GetWaitForLaunch(self):
        ...

    def SetWaitForLaunch(self, *args):
        ...

    def GetIgnoreExisting(self):
        ...

    def SetIgnoreExisting(self, b):
        ...

    def GetResumeCount(self):
        ...

    def SetResumeCount(self, c):
        ...

    def GetProcessPluginName(self):
        ...

    def SetProcessPluginName(self, plugin_name):
        ...

    def GetUserID(self):
        ...

    def GetGroupID(self):
        ...

    def UserIDIsValid(self):
        ...

    def GroupIDIsValid(self):
        ...

    def SetUserID(self, uid):
        ...

    def SetGroupID(self, gid):
        ...

    def GetEffectiveUserID(self):
        ...

    def GetEffectiveGroupID(self):
        ...

    def EffectiveUserIDIsValid(self):
        ...

    def EffectiveGroupIDIsValid(self):
        ...

    def SetEffectiveUserID(self, uid):
        ...

    def SetEffectiveGroupID(self, gid):
        ...

    def GetParentProcessID(self):
        ...

    def SetParentProcessID(self, pid):
        ...

    def ParentProcessIDIsValid(self):
        ...

    def GetListener(self):
        ...

    def SetListener(self, listener):
        ...

    def GetShadowListener(self):
        ...

    def SetShadowListener(self, listener):
        ...

    def GetScriptedProcessClassName(self):
        ...

    def SetScriptedProcessClassName(self, class_name):
        ...

    def GetScriptedProcessDictionary(self):
        ...

    def SetScriptedProcessDictionary(self, dict):
        ...


class SBBlock:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def IsInlined(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetInlinedName(self):
        ...

    def GetInlinedCallSiteFile(self):
        ...

    def GetInlinedCallSiteLine(self):
        ...

    def GetInlinedCallSiteColumn(self):
        ...

    def GetParent(self):
        ...

    def GetSibling(self):
        ...

    def GetFirstChild(self):
        ...

    def GetNumRanges(self):
        ...

    def GetRangeStartAddress(self, idx):
        ...

    def GetRangeEndAddress(self, idx):
        ...

    def GetRangeIndexForBlockAddress(self, block_addr):
        ...

    def GetVariables(self, *args):
        ...

    def GetContainingInlinedBlock(self):
        ...

    def GetDescription(self, description):
        ...

    def get_range_at_index(self, idx):
        ...

    class ranges_access:
        sbblock: Incomplete

        def __init__(self, sbblock) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_ranges_access_object(self):
        ...

    ranges_array: Incomplete

    def get_ranges_array(self):
        ...

    def get_call_site(self):
        ...

    parent: Incomplete
    first_child: Incomplete
    call_site: Incomplete
    sibling: Incomplete
    name: Incomplete
    inlined_block: Incomplete
    range: Incomplete
    ranges: Incomplete
    num_ranges: Incomplete


class SBBreakpoint:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetID(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def ClearAllBreakpointSites(self):
        ...

    def GetTarget(self):
        ...

    def FindLocationByAddress(self, vm_addr):
        ...

    def FindLocationIDByAddress(self, vm_addr):
        ...

    def FindLocationByID(self, bp_loc_id):
        ...

    def GetLocationAtIndex(self, index):
        ...

    def SetEnabled(self, enable):
        ...

    def IsEnabled(self):
        ...

    def SetOneShot(self, one_shot):
        ...

    def IsOneShot(self):
        ...

    def IsInternal(self):
        ...

    def GetHitCount(self):
        ...

    def SetIgnoreCount(self, count):
        ...

    def GetIgnoreCount(self):
        ...

    def SetCondition(self, condition):
        ...

    def GetCondition(self):
        ...

    def SetAutoContinue(self, auto_continue):
        ...

    def GetAutoContinue(self):
        ...

    def SetThreadID(self, sb_thread_id):
        ...

    def GetThreadID(self):
        ...

    def SetThreadIndex(self, index):
        ...

    def GetThreadIndex(self):
        ...

    def SetThreadName(self, thread_name):
        ...

    def GetThreadName(self):
        ...

    def SetQueueName(self, queue_name):
        ...

    def GetQueueName(self):
        ...

    def SetScriptCallbackFunction(self, *args):
        ...

    def SetCommandLineCommands(self, commands):
        ...

    def GetCommandLineCommands(self, commands):
        ...

    def SetScriptCallbackBody(self, script_body_text):
        ...

    def AddName(self, new_name):
        ...

    def AddNameWithErrorHandling(self, new_name):
        ...

    def RemoveName(self, name_to_remove):
        ...

    def MatchesName(self, name):
        ...

    def GetNames(self, names):
        ...

    def GetNumResolvedLocations(self):
        ...

    def GetNumLocations(self):
        ...

    def GetDescription(self, *args):
        ...

    @staticmethod
    def EventIsBreakpointEvent(event):
        ...

    @staticmethod
    def GetBreakpointEventTypeFromEvent(event):
        ...

    @staticmethod
    def GetBreakpointFromEvent(event):
        ...

    @staticmethod
    def GetBreakpointLocationAtIndexFromEvent(event, loc_idx):
        ...

    @staticmethod
    def GetNumBreakpointLocationsFromEvent(event_sp):
        ...

    def IsHardware(self):
        ...

    def AddLocation(self, address):
        ...

    def SerializeToStructuredData(self):
        ...

    class locations_access:
        sbbreakpoint: Incomplete

        def __init__(self, sbbreakpoint) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_locations_access_object(self):
        ...

    def get_breakpoint_location_list(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    locations: Incomplete
    location: Incomplete
    id: Incomplete
    enabled: Incomplete
    one_shot: Incomplete
    num_locations: Incomplete


class SBBreakpointList:
    thisown: Incomplete

    def __init__(self, target) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetSize(self):
        ...

    def GetBreakpointAtIndex(self, idx):
        ...

    def FindBreakpointByID(self, arg2):
        ...

    def Append(self, sb_bkpt):
        ...

    def AppendIfUnique(self, sb_bkpt):
        ...

    def AppendByID(self, id):
        ...

    def Clear(self):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...


class SBBreakpointLocation:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetID(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetAddress(self):
        ...

    def GetLoadAddress(self):
        ...

    def SetEnabled(self, enabled):
        ...

    def IsEnabled(self):
        ...

    def GetHitCount(self):
        ...

    def GetIgnoreCount(self):
        ...

    def SetIgnoreCount(self, n):
        ...

    def SetCondition(self, condition):
        ...

    def GetCondition(self):
        ...

    def SetAutoContinue(self, auto_continue):
        ...

    def GetAutoContinue(self):
        ...

    def SetScriptCallbackFunction(self, *args):
        ...

    def SetScriptCallbackBody(self, script_body_text):
        ...

    def SetCommandLineCommands(self, commands):
        ...

    def GetCommandLineCommands(self, commands):
        ...

    def SetThreadID(self, sb_thread_id):
        ...

    def GetThreadID(self):
        ...

    def SetThreadIndex(self, index):
        ...

    def GetThreadIndex(self):
        ...

    def SetThreadName(self, thread_name):
        ...

    def GetThreadName(self):
        ...

    def SetQueueName(self, queue_name):
        ...

    def GetQueueName(self):
        ...

    def IsResolved(self):
        ...

    def GetDescription(self, description, level):
        ...

    def GetBreakpoint(self):
        ...


class SBBreakpointName:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetName(self):
        ...

    def SetEnabled(self, enable):
        ...

    def IsEnabled(self):
        ...

    def SetOneShot(self, one_shot):
        ...

    def IsOneShot(self):
        ...

    def SetIgnoreCount(self, count):
        ...

    def GetIgnoreCount(self):
        ...

    def SetCondition(self, condition):
        ...

    def GetCondition(self):
        ...

    def SetAutoContinue(self, auto_continue):
        ...

    def GetAutoContinue(self):
        ...

    def SetThreadID(self, sb_thread_id):
        ...

    def GetThreadID(self):
        ...

    def SetThreadIndex(self, index):
        ...

    def GetThreadIndex(self):
        ...

    def SetThreadName(self, thread_name):
        ...

    def GetThreadName(self):
        ...

    def SetQueueName(self, queue_name):
        ...

    def GetQueueName(self):
        ...

    def SetScriptCallbackFunction(self, *args):
        ...

    def SetCommandLineCommands(self, commands):
        ...

    def GetCommandLineCommands(self, commands):
        ...

    def SetScriptCallbackBody(self, script_body_text):
        ...

    def GetHelpString(self):
        ...

    def SetHelpString(self, help_string):
        ...

    def GetAllowList(self):
        ...

    def SetAllowList(self, value):
        ...

    def GetAllowDelete(self):
        ...

    def SetAllowDelete(self, value):
        ...

    def GetAllowDisable(self):
        ...

    def SetAllowDisable(self, value):
        ...

    def GetDescription(self, description):
        ...


class SBBroadcaster:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def BroadcastEventByType(self, event_type, unique: bool = False):
        ...

    def BroadcastEvent(self, event, unique: bool = False):
        ...

    def AddInitialEventsToListener(self, listener, requested_events):
        ...

    def AddListener(self, listener, event_mask):
        ...

    def GetName(self):
        ...

    def EventTypeHasListeners(self, event_type):
        ...

    def RemoveListener(self, *args):
        ...

    def __lt__(self, rhs):
        ...


class SBCommandInterpreter:
    thisown: Incomplete
    eBroadcastBitThreadShouldExit: Incomplete
    eBroadcastBitResetPrompt: Incomplete
    eBroadcastBitQuitCommandReceived: Incomplete
    eBroadcastBitAsynchronousOutputData: Incomplete
    eBroadcastBitAsynchronousErrorData: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    @staticmethod
    def GetArgumentTypeAsCString(arg_type):
        ...

    @staticmethod
    def GetArgumentDescriptionAsCString(arg_type):
        ...

    @staticmethod
    def EventIsCommandInterpreterEvent(event):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def CommandExists(self, cmd):
        ...

    def UserCommandExists(self, cmd):
        ...

    def AliasExists(self, cmd):
        ...

    def GetBroadcaster(self):
        ...

    @staticmethod
    def GetBroadcasterClass():
        ...

    def HasCommands(self):
        ...

    def HasAliases(self):
        ...

    def HasAliasOptions(self):
        ...

    def IsInteractive(self):
        ...

    def GetProcess(self):
        ...

    def GetDebugger(self):
        ...

    def SourceInitFileInHomeDirectory(self, *args):
        ...

    def SourceInitFileInCurrentWorkingDirectory(self, result):
        ...

    def HandleCommand(self, *args: Any):
        ...

    def HandleCommandsFromFile(self, file, override_context, options, result):
        ...

    def HandleCompletion(
        self,
        current_line,
        cursor_pos,
        match_start_point,
        max_return_elements,
        matches,
    ):
        ...

    def HandleCompletionWithDescriptions(
        self,
        current_line,
        cursor_pos,
        match_start_point,
        max_return_elements,
        matches,
        descriptions,
    ):
        ...

    def WasInterrupted(self):
        ...

    def InterruptCommand(self):
        ...

    def IsActive(self):
        ...

    def GetIOHandlerControlSequence(self, ch):
        ...

    def GetPromptOnQuit(self):
        ...

    def SetPromptOnQuit(self, b):
        ...

    def AllowExitCodeOnQuit(self, allow):
        ...

    def HasCustomQuitExitCode(self):
        ...

    def GetQuitStatus(self):
        ...

    def ResolveCommand(self, command_line, result):
        ...


class SBCommandInterpreterRunOptions:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetStopOnContinue(self):
        ...

    def SetStopOnContinue(self, arg2):
        ...

    def GetStopOnError(self):
        ...

    def SetStopOnError(self, arg2):
        ...

    def GetStopOnCrash(self):
        ...

    def SetStopOnCrash(self, arg2):
        ...

    def GetEchoCommands(self):
        ...

    def SetEchoCommands(self, arg2):
        ...

    def GetEchoCommentCommands(self):
        ...

    def SetEchoCommentCommands(self, echo):
        ...

    def GetPrintResults(self):
        ...

    def SetPrintResults(self, arg2):
        ...

    def GetPrintErrors(self):
        ...

    def SetPrintErrors(self, arg2):
        ...

    def GetAddToHistory(self):
        ...

    def SetAddToHistory(self, arg2):
        ...

    def GetAutoHandleEvents(self):
        ...

    def SetAutoHandleEvents(self, arg2):
        ...

    def GetSpawnThread(self):
        ...

    def SetSpawnThread(self, arg2):
        ...


class SBCommandReturnObject:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def PutOutput(self, *args):
        ...

    def GetOutputSize(self):
        ...

    def GetErrorSize(self):
        ...

    def PutError(self, *args):
        ...

    def Clear(self):
        ...

    def GetStatus(self):
        ...

    def SetStatus(self, status: int):
        ...

    def Succeeded(self):
        ...

    def HasResult(self):
        ...

    def AppendMessage(self, message):
        ...

    def AppendWarning(self, message):
        ...

    def GetDescription(self, description):
        ...

    def PutCString(self, string: str) -> None:
        ...

    def GetOutput(self, *args):
        ...

    def GetError(self, *args):
        ...

    def SetError(self, *args):
        ...

    def SetImmediateOutputFile(self, *args):
        ...

    def SetImmediateErrorFile(self, *args):
        ...

    def Print(self, str):
        ...

    def write(self, str):
        ...

    def flush(self):
        ...


class SBCommunication:
    thisown: Incomplete
    eBroadcastBitDisconnected: Incomplete
    eBroadcastBitReadThreadGotBytes: Incomplete
    eBroadcastBitReadThreadDidExit: Incomplete
    eBroadcastBitReadThreadShouldExit: Incomplete
    eBroadcastBitPacketAvailable: Incomplete
    eAllEventBits: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetBroadcaster(self):
        ...

    @staticmethod
    def GetBroadcasterClass():
        ...

    def AdoptFileDesriptor(self, fd, owns_fd):
        ...

    def Connect(self, url):
        ...

    def Disconnect(self):
        ...

    def IsConnected(self):
        ...

    def GetCloseOnEOF(self):
        ...

    def SetCloseOnEOF(self, b):
        ...

    def Read(self, dst, dst_len, timeout_usec, status):
        ...

    def Write(self, src, src_len, status):
        ...

    def ReadThreadStart(self):
        ...

    def ReadThreadStop(self):
        ...

    def ReadThreadIsRunning(self):
        ...

    def SetReadThreadBytesReceivedCallback(self, callback, callback_baton):
        ...


class SBCompileUnit:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetFileSpec(self):
        ...

    def GetNumLineEntries(self):
        ...

    def GetLineEntryAtIndex(self, idx):
        ...

    def FindLineEntryIndex(self, *args):
        ...

    def GetSupportFileAtIndex(self, idx):
        ...

    def GetNumSupportFiles(self):
        ...

    def FindSupportFileIndex(self, start_idx, sb_file, full):
        ...

    def GetTypes(self, *args):
        ...

    def GetLanguage(self):
        ...

    def GetDescription(self, description):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    file: Incomplete
    num_line_entries: Incomplete


class SBData:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetAddressByteSize(self):
        ...

    def SetAddressByteSize(self, addr_byte_size):
        ...

    def Clear(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetByteSize(self):
        ...

    def GetByteOrder(self):
        ...

    def SetByteOrder(self, endian):
        ...

    def GetFloat(self, error, offset):
        ...

    def GetDouble(self, error, offset):
        ...

    def GetLongDouble(self, error, offset):
        ...

    def GetAddress(self, error, offset):
        ...

    def GetUnsignedInt8(self, error, offset):
        ...

    def GetUnsignedInt16(self, error, offset):
        ...

    def GetUnsignedInt32(self, error, offset):
        ...

    def GetUnsignedInt64(self, error, offset):
        ...

    def GetSignedInt8(self, error, offset):
        ...

    def GetSignedInt16(self, error, offset):
        ...

    def GetSignedInt32(self, error, offset):
        ...

    def GetSignedInt64(self, error, offset):
        ...

    def GetString(self, error, offset):
        ...

    def ReadRawData(self, error, offset, buf):
        ...

    def GetDescription(self, *args):
        ...

    def SetData(self, error, buf, endian, addr_size):
        ...

    def SetDataWithOwnership(self, error, buf, endian, addr_size):
        ...

    def Append(self, rhs):
        ...

    @staticmethod
    def CreateDataFromCString(endian, addr_byte_size, data):
        ...

    @staticmethod
    def CreateDataFromUInt64Array(endian, addr_byte_size, array):
        ...

    @staticmethod
    def CreateDataFromUInt32Array(endian, addr_byte_size, array):
        ...

    @staticmethod
    def CreateDataFromSInt64Array(endian, addr_byte_size, array):
        ...

    @staticmethod
    def CreateDataFromSInt32Array(endian, addr_byte_size, array):
        ...

    @staticmethod
    def CreateDataFromDoubleArray(endian, addr_byte_size, array):
        ...

    def SetDataFromCString(self, data):
        ...

    def SetDataFromUInt64Array(self, array):
        ...

    def SetDataFromUInt32Array(self, array):
        ...

    def SetDataFromSInt64Array(self, array):
        ...

    def SetDataFromSInt32Array(self, array):
        ...

    def SetDataFromDoubleArray(self, array):
        ...

    def __len__(self) -> int:
        ...

    class read_data_helper:
        sbdata: Incomplete
        readerfunc: Incomplete
        item_size: Incomplete

        def __init__(self, sbdata, readerfunc, item_size) -> None:
            ...

        def __getitem__(self, key):
            ...

        def __len__(self) -> int:
            ...

        def all(self):
            ...

    @classmethod
    def CreateDataFromInt(
        cls,
        value,
        size: Incomplete = None,
        target: Incomplete = None,
        ptr_size: Incomplete = None,
        endian: Incomplete = None,
    ):
        ...

    uint8: Incomplete
    uint16: Incomplete
    uint32: Incomplete
    uint64: Incomplete
    sint8: Incomplete
    sint16: Incomplete
    sint32: Incomplete
    sint64: Incomplete
    float: Incomplete
    double: Incomplete
    uint8s: Incomplete
    uint16s: Incomplete
    uint32s: Incomplete
    uint64s: Incomplete
    sint8s: Incomplete
    sint16s: Incomplete
    sint32s: Incomplete
    sint64s: Incomplete
    floats: Incomplete
    doubles: Incomplete
    byte_order: Incomplete
    size: Incomplete


class SBDebugger:
    thisown: Incomplete
    eBroadcastBitProgress: Incomplete
    eBroadcastBitWarning: Incomplete
    eBroadcastBitError: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    @staticmethod
    def GetBroadcasterClass():
        ...

    def GetBroadcaster(self):
        ...

    @staticmethod
    def GetProgressFromEvent(event):
        ...

    @staticmethod
    def GetProgressDataFromEvent(event):
        ...

    @staticmethod
    def GetDiagnosticFromEvent(event):
        ...

    @staticmethod
    def Initialize():
        ...

    @staticmethod
    def InitializeWithErrorHandling():
        ...

    @staticmethod
    def PrintStackTraceOnError():
        ...

    @staticmethod
    def PrintDiagnosticsOnError():
        ...

    @staticmethod
    def Terminate():
        ...

    @staticmethod
    def Create(*args):
        ...

    @staticmethod
    def Destroy(debugger):
        ...

    @staticmethod
    def MemoryPressureDetected():
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetSetting(self, setting: Incomplete = None):
        ...

    def SetAsync(self, b: bool):
        ...

    def GetAsync(self):
        ...

    def SkipLLDBInitFiles(self, b):
        ...

    def SkipAppInitFiles(self, b):
        ...

    def SetInputString(self, data):
        ...

    def SetInputFile(self, *args):
        ...

    def SetOutputFile(self, *args):
        ...

    def SetErrorFile(self, *args):
        ...

    def GetInputFile(self):
        ...

    def GetOutputFile(self):
        ...

    def GetErrorFile(self):
        ...

    def SaveInputTerminalState(self):
        ...

    def RestoreInputTerminalState(self):
        ...

    def GetCommandInterpreter(self) -> SBCommandInterpreter:
        ...

    def HandleCommand(self, command: Any):
        ...

    def RequestInterrupt(self):
        ...

    def CancelInterruptRequest(self):
        ...

    def InterruptRequested(self):
        ...

    def GetListener(self):
        ...

    def HandleProcessEvent(self, *args):
        ...

    def CreateTargetWithFileAndTargetTriple(self, filename, target_triple):
        ...

    def CreateTargetWithFileAndArch(self, filename, archname):
        ...

    def CreateTarget(self, *args: Any) -> "SBTarget":
        ...

    def GetDummyTarget(self):
        ...

    def DeleteTarget(self, target):
        ...

    def GetTargetAtIndex(self, idx):
        ...

    def GetIndexOfTarget(self, target):
        ...

    def FindTargetWithProcessID(self, pid):
        ...

    def FindTargetWithFileAndArch(self, filename, arch):
        ...

    def GetNumTargets(self):
        ...

    def GetSelectedTarget(self):
        ...

    def SetSelectedTarget(self, target):
        ...

    def GetSelectedPlatform(self):
        ...

    def SetSelectedPlatform(self, platform):
        ...

    def GetNumPlatforms(self):
        ...

    def GetPlatformAtIndex(self, idx):
        ...

    def GetNumAvailablePlatforms(self):
        ...

    def GetAvailablePlatformInfoAtIndex(self, idx):
        ...

    def GetSourceManager(self):
        ...

    def SetCurrentPlatform(self, platform_name):
        ...

    def SetCurrentPlatformSDKRoot(self, sysroot):
        ...

    def SetUseExternalEditor(self, input):
        ...

    def GetUseExternalEditor(self):
        ...

    def SetUseColor(self, use_color):
        ...

    def GetUseColor(self):
        ...

    def SetUseSourceCache(self, use_source_cache):
        ...

    def GetUseSourceCache(self):
        ...

    @staticmethod
    def GetDefaultArchitecture(arch_name, arch_name_len):
        ...

    @staticmethod
    def SetDefaultArchitecture(arch_name):
        ...

    def GetScriptingLanguage(self, script_language_name):
        ...

    def GetScriptInterpreterInfo(self, arg2):
        ...

    @staticmethod
    def GetVersionString():
        ...

    @staticmethod
    def StateAsCString(state):
        ...

    @staticmethod
    def GetBuildConfiguration():
        ...

    @staticmethod
    def StateIsRunningState(state):
        ...

    @staticmethod
    def StateIsStoppedState(state):
        ...

    def EnableLog(self, channel, categories):
        ...

    def SetLoggingCallback(self, log_callback):
        ...

    def SetDestroyCallback(self, destroy_callback):
        ...

    def DispatchInput(self, data):
        ...

    def DispatchInputInterrupt(self):
        ...

    def DispatchInputEndOfFile(self):
        ...

    def GetInstanceName(self):
        ...

    @staticmethod
    def FindDebuggerWithID(id):
        ...

    @staticmethod
    def SetInternalVariable(var_name, value, debugger_instance_name):
        ...

    @staticmethod
    def GetInternalVariableValue(var_name, debugger_instance_name):
        ...

    def GetDescription(self, description):
        ...

    def GetTerminalWidth(self):
        ...

    def SetTerminalWidth(self, term_width):
        ...

    def GetID(self):
        ...

    def GetPrompt(self):
        ...

    def SetPrompt(self, prompt):
        ...

    def GetReproducerPath(self):
        ...

    def GetScriptLanguage(self):
        ...

    def SetScriptLanguage(self, script_lang):
        ...

    def GetREPLLanguage(self):
        ...

    def SetREPLLanguage(self, repl_lang):
        ...

    def GetCloseInputOnEOF(self):
        ...

    def SetCloseInputOnEOF(self, b):
        ...

    def GetCategory(self, *args):
        ...

    def CreateCategory(self, category_name):
        ...

    def DeleteCategory(self, category_name):
        ...

    def GetNumCategories(self):
        ...

    def GetCategoryAtIndex(self, arg2):
        ...

    def GetDefaultCategory(self):
        ...

    def GetFormatForType(self, arg2):
        ...

    def GetSummaryForType(self, arg2):
        ...

    def GetFilterForType(self, arg2):
        ...

    def GetSyntheticForType(self, arg2):
        ...

    def RunCommandInterpreter(
        self,
        auto_handle_events,
        spawn_thread,
        options,
        num_errors,
        quit_requested,
        stopped_for_crash,
    ):
        ...

    def RunREPL(self, language, repl_options):
        ...

    def LoadTraceFromFile(self, error, trace_description_file):
        ...

    def SetOutputFileHandle(self, file, transfer_ownership) -> None:
        ...

    def SetInputFileHandle(self, file, transfer_ownership) -> None:
        ...

    def SetErrorFileHandle(self, file, transfer_ownership) -> None:
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def GetInputFileHandle(self):
        ...

    def GetOutputFileHandle(self):
        ...

    def GetErrorFileHandle(self):
        ...


class SBDeclaration:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetFileSpec(self):
        ...

    def GetLine(self):
        ...

    def GetColumn(self):
        ...

    def SetFileSpec(self, filespec):
        ...

    def SetLine(self, line):
        ...

    def SetColumn(self, column):
        ...

    def GetDescription(self, description):
        ...

    file: Incomplete
    line: Incomplete
    column: Incomplete


class SBError:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetCString(self):
        ...

    def Clear(self):
        ...

    def Fail(self):
        ...

    def Success(self):
        ...

    def GetError(self):
        ...

    def GetType(self):
        ...

    def SetError(self, err, type):
        ...

    def SetErrorToErrno(self):
        ...

    def SetErrorToGenericError(self):
        ...

    def SetErrorString(self, err_str):
        ...

    def SetErrorStringWithFormat(
        self,
        format,
        str1: Incomplete = None,
        str2: Incomplete = None,
        str3: Incomplete = None,
    ):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetDescription(self, description):
        ...

    def __int__(self) -> int:
        ...

    value: Incomplete
    fail: Incomplete
    success: Incomplete
    description: Incomplete
    type: Incomplete


class SBEnvironment:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def Get(self, name):
        ...

    def GetNumValues(self):
        ...

    def GetNameAtIndex(self, index):
        ...

    def GetValueAtIndex(self, index):
        ...

    def GetEntries(self):
        ...

    def PutEntry(self, name_and_value):
        ...

    def SetEntries(self, entries, append):
        ...

    def Set(self, name, value, overwrite):
        ...

    def Unset(self, name):
        ...

    def Clear(self):
        ...


class SBEvent:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetDataFlavor(self):
        ...

    def GetType(self):
        ...

    def GetBroadcaster(self):
        ...

    def GetBroadcasterClass(self):
        ...

    def BroadcasterMatchesRef(self, broadcaster):
        ...

    def Clear(self):
        ...

    @staticmethod
    def GetCStringFromEvent(event):
        ...

    def GetDescription(self, *args):
        ...


class SBExecutionContext:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetTarget(self):
        ...

    def GetProcess(self):
        ...

    def GetThread(self):
        ...

    def GetFrame(self):
        ...

    target: Incomplete
    process: Incomplete
    thread: Incomplete
    frame: Incomplete


class SBExpressionOptions:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetCoerceResultToId(self):
        ...

    def SetCoerceResultToId(self, coerce: bool = True):
        ...

    def GetUnwindOnError(self):
        ...

    def SetUnwindOnError(self, unwind: bool = True):
        ...

    def GetIgnoreBreakpoints(self):
        ...

    def SetIgnoreBreakpoints(self, ignore: bool = True):
        ...

    def GetFetchDynamicValue(self):
        ...

    def SetFetchDynamicValue(self, *args):
        ...

    def GetTimeoutInMicroSeconds(self):
        ...

    def SetTimeoutInMicroSeconds(self, timeout: int = 0):
        ...

    def GetOneThreadTimeoutInMicroSeconds(self):
        ...

    def SetOneThreadTimeoutInMicroSeconds(self, timeout: int = 0):
        ...

    def GetTryAllThreads(self):
        ...

    def SetTryAllThreads(self, run_others: bool = True):
        ...

    def GetStopOthers(self):
        ...

    def SetStopOthers(self, stop_others: bool = True):
        ...

    def GetTrapExceptions(self):
        ...

    def SetTrapExceptions(self, trap_exceptions: bool = True):
        ...

    def SetLanguage(self, language):
        ...

    def GetGenerateDebugInfo(self):
        ...

    def SetGenerateDebugInfo(self, b: bool = True):
        ...

    def GetSuppressPersistentResult(self):
        ...

    def SetSuppressPersistentResult(self, b: bool = False):
        ...

    def GetPrefix(self):
        ...

    def SetPrefix(self, prefix):
        ...

    def SetAutoApplyFixIts(self, b: bool = True):
        ...

    def GetAutoApplyFixIts(self):
        ...

    def SetRetriesWithFixIts(self, retries):
        ...

    def GetRetriesWithFixIts(self):
        ...

    def GetTopLevel(self):
        ...

    def SetTopLevel(self, b: bool = True):
        ...

    def GetAllowJIT(self):
        ...

    def SetAllowJIT(self, allow):
        ...


class SBFile:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def Read(self, buf):
        ...

    def Write(self, buf):
        ...

    def Flush(self):
        ...

    def IsValid(self):
        ...

    def Close(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def GetFile(self):
        ...

    @staticmethod
    def MakeBorrowed(BORROWED):
        ...

    @staticmethod
    def MakeForcingIOMethods(FORCE_IO_METHODS):
        ...

    @staticmethod
    def MakeBorrowedForcingIOMethods(BORROWED_FORCE_IO_METHODS):
        ...

    @classmethod
    def Create(cls, file, borrow: bool = False, force_io_methods: bool = False):
        ...


class SBFileSpec:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Exists(self):
        ...

    def ResolveExecutableLocation(self):
        ...

    def GetFilename(self):
        ...

    def GetDirectory(self):
        ...

    def SetFilename(self, filename):
        ...

    def SetDirectory(self, directory):
        ...

    def GetPath(self, dst_path, dst_len):
        ...

    @staticmethod
    def ResolvePath(src_path, dst_path, dst_len):
        ...

    def GetDescription(self, description):
        ...

    def AppendPathComponent(self, file_or_directory):
        ...

    fullpath: Incomplete
    basename: Incomplete
    dirname: Incomplete
    exists: Incomplete


class SBFileSpecList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetSize(self):
        ...

    def GetDescription(self, description):
        ...

    def Append(self, sb_file):
        ...

    def AppendIfUnique(self, sb_file):
        ...

    def Clear(self):
        ...

    def FindFileIndex(self, idx, sb_file, full):
        ...

    def GetFileSpecAtIndex(self, idx):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...


class SBFormat:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__


class SBFrame:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def IsEqual(self, that):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetFrameID(self):
        ...

    def GetCFA(self):
        ...

    def GetPC(self):
        ...

    def SetPC(self, new_pc):
        ...

    def GetSP(self):
        ...

    def GetFP(self):
        ...

    def GetPCAddress(self):
        ...

    def GetSymbolContext(self, resolve_scope):
        ...

    def GetModule(self):
        ...

    def GetCompileUnit(self):
        ...

    def GetFunction(self):
        ...

    def GetSymbol(self):
        ...

    def GetBlock(self):
        ...

    def GetDisplayFunctionName(self):
        ...

    def GetFunctionName(self, *args):
        ...

    def GuessLanguage(self):
        ...

    def IsInlined(self, *args):
        ...

    def IsArtificial(self, *args):
        ...

    def EvaluateExpression(self, *args: Any) -> "SBValue":
        ...

    def GetFrameBlock(self):
        ...

    def GetLineEntry(self) -> "SBLineEntry":
        ...

    def GetThread(self):
        ...

    def Disassemble(self):
        ...

    def Clear(self):
        ...

    def GetVariables(self, *args):
        ...

    def GetRegisters(self):
        ...

    def FindRegister(self, name):
        ...

    def FindVariable(self, *args: Any) -> "SBValue":
        ...

    def GetValueForVariablePath(self, *args: Any) -> "SBValue":
        ...

    def FindValue(self, *args):
        ...

    def GetDescription(self, description):
        ...

    def GetDescriptionWithFormat(self, format, output):
        ...

    def get_instructions_from_current_target(self):
        ...

    addr: Incomplete
    end_addr: Incomplete
    block: Incomplete
    instructions: Incomplete
    mangled: Incomplete
    name: Incomplete
    prologue_size: Incomplete
    type: Incomplete


class SBHostOS:
    thisown: Incomplete

    @staticmethod
    def GetProgramFileSpec():
        ...

    @staticmethod
    def GetLLDBPythonPath():
        ...

    @staticmethod
    def GetLLDBPath(path_type):
        ...

    @staticmethod
    def GetUserHomeDirectory():
        ...

    @staticmethod
    def ThreadCreated(name):
        ...

    @staticmethod
    def ThreadCreate(name, thread_function, thread_arg, err):
        ...

    @staticmethod
    def ThreadCancel(thread, err):
        ...

    @staticmethod
    def ThreadDetach(thread, err):
        ...

    @staticmethod
    def ThreadJoin(thread, result, err):
        ...

    def __init__(self) -> None:
        ...

    __swig_destroy__: Incomplete


class SBInstruction:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetAddress(self):
        ...

    def GetMnemonic(self, target):
        ...

    def GetOperands(self, target):
        ...

    def GetComment(self, target):
        ...

    def GetControlFlowKind(self, target):
        ...

    def GetData(self, target):
        ...

    def GetByteSize(self):
        ...

    def DoesBranch(self):
        ...

    def HasDelaySlot(self):
        ...

    def CanSetBreakpoint(self):
        ...

    def Print(self, *args):
        ...

    def GetDescription(self, description):
        ...

    def EmulateWithFrame(self, frame, evaluate_options):
        ...

    def DumpEmulation(self, triple):
        ...

    def TestEmulation(self, output_stream, test_file):
        ...

    def __hex__(self):
        ...

    def __len__(self) -> int:
        ...

    def __mnemonic_property__(self):
        ...

    def __operands_property__(self):
        ...

    def __comment_property__(self):
        ...

    def __file_addr_property__(self):
        ...

    def __load_adrr_property__(self):
        ...

    mnemonic: Incomplete
    operands: Incomplete
    comment: Incomplete
    addr: Incomplete
    size: Incomplete
    is_branch: Incomplete


class SBInstructionList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetSize(self):
        ...

    def GetInstructionAtIndex(self, idx):
        ...

    def GetInstructionsCount(self, start, end, canSetBreakpoint: bool = False):
        ...

    def Clear(self):
        ...

    def AppendInstruction(self, inst):
        ...

    def Print(self, *args):
        ...

    def GetDescription(self, description):
        ...

    def DumpEmulationForAllInstructions(self, triple):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, key):
        ...


class SBLanguageRuntime:
    thisown: Incomplete

    @staticmethod
    def GetLanguageTypeFromString(string):
        ...

    @staticmethod
    def GetNameForLanguageType(language):
        ...

    def __init__(self) -> None:
        ...

    __swig_destroy__: Incomplete


class SBLaunchInfo:
    thisown: Incomplete

    def __init__(self, argv) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetProcessID(self):
        ...

    def GetUserID(self):
        ...

    def GetGroupID(self):
        ...

    def UserIDIsValid(self):
        ...

    def GroupIDIsValid(self):
        ...

    def SetUserID(self, uid):
        ...

    def SetGroupID(self, gid):
        ...

    def GetExecutableFile(self):
        ...

    def SetExecutableFile(self, exe_file, add_as_first_arg):
        ...

    def GetListener(self):
        ...

    def SetListener(self, listener):
        ...

    def GetShadowListener(self):
        ...

    def SetShadowListener(self, listener):
        ...

    def GetNumArguments(self):
        ...

    def GetArgumentAtIndex(self, idx):
        ...

    def SetArguments(self, argv, append):
        ...

    def GetNumEnvironmentEntries(self):
        ...

    def GetEnvironmentEntryAtIndex(self, idx):
        ...

    def SetEnvironmentEntries(self, envp, append):
        ...

    def SetEnvironment(self, env, append):
        ...

    def GetEnvironment(self):
        ...

    def Clear(self):
        ...

    def GetWorkingDirectory(self):
        ...

    def SetWorkingDirectory(self, working_dir):
        ...

    def GetLaunchFlags(self):
        ...

    def SetLaunchFlags(self, flags):
        ...

    def GetProcessPluginName(self):
        ...

    def SetProcessPluginName(self, plugin_name):
        ...

    def GetShell(self):
        ...

    def SetShell(self, path):
        ...

    def GetShellExpandArguments(self):
        ...

    def SetShellExpandArguments(self, expand):
        ...

    def GetResumeCount(self):
        ...

    def SetResumeCount(self, c):
        ...

    def AddCloseFileAction(self, fd):
        ...

    def AddDuplicateFileAction(self, fd, dup_fd):
        ...

    def AddOpenFileAction(self, fd, path, read, write):
        ...

    def AddSuppressFileAction(self, fd, read, write):
        ...

    def SetLaunchEventData(self, data):
        ...

    def GetLaunchEventData(self):
        ...

    def GetDetachOnError(self):
        ...

    def SetDetachOnError(self, enable):
        ...

    def GetScriptedProcessClassName(self):
        ...

    def SetScriptedProcessClassName(self, class_name):
        ...

    def GetScriptedProcessDictionary(self):
        ...

    def SetScriptedProcessDictionary(self, dict):
        ...


class SBLineEntry:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetStartAddress(self):
        ...

    def GetEndAddress(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetFileSpec(self):
        ...

    def GetLine(self) -> int:
        ...

    def GetColumn(self) -> int:
        ...

    def SetFileSpec(self, filespec):
        ...

    def SetLine(self, line):
        ...

    def SetColumn(self, column):
        ...

    def GetDescription(self, description):
        ...

    def __int__(self) -> int:
        ...

    def __hex__(self):
        ...

    file: Incomplete
    line: Incomplete
    column: Incomplete
    addr: Incomplete
    end_addr: Incomplete


class SBListener:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def AddEvent(self, event):
        ...

    def Clear(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def StartListeningForEventClass(
        self, debugger, broadcaster_class, event_mask
    ):
        ...

    def StopListeningForEventClass(
        self, debugger, broadcaster_class, event_mask
    ):
        ...

    def StartListeningForEvents(self, broadcaster, event_mask):
        ...

    def StopListeningForEvents(self, broadcaster, event_mask):
        ...

    def WaitForEvent(self, num_seconds, event):
        ...

    def WaitForEventForBroadcaster(self, num_seconds, broadcaster, sb_event):
        ...

    def WaitForEventForBroadcasterWithType(
        self, num_seconds, broadcaster, event_type_mask, sb_event
    ):
        ...

    def PeekAtNextEvent(self, sb_event):
        ...

    def PeekAtNextEventForBroadcaster(self, broadcaster, sb_event):
        ...

    def PeekAtNextEventForBroadcasterWithType(
        self, broadcaster, event_type_mask, sb_event
    ):
        ...

    def GetNextEvent(self, sb_event):
        ...

    def GetNextEventForBroadcaster(self, broadcaster, sb_event):
        ...

    def GetNextEventForBroadcasterWithType(
        self, broadcaster, event_type_mask, sb_event
    ):
        ...

    def HandleBroadcastEvent(self, event):
        ...


class SBMemoryRegionInfo:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def Clear(self):
        ...

    def GetRegionBase(self):
        ...

    def GetRegionEnd(self):
        ...

    def IsReadable(self):
        ...

    def IsWritable(self):
        ...

    def IsExecutable(self):
        ...

    def IsMapped(self):
        ...

    def GetName(self):
        ...

    def HasDirtyMemoryPageList(self):
        ...

    def GetNumDirtyPages(self):
        ...

    def GetDirtyPageAddressAtIndex(self, idx):
        ...

    def GetPageSize(self):
        ...

    def GetDescription(self, description):
        ...

    def __hex__(self):
        ...

    def __len__(self) -> int:
        ...


class SBMemoryRegionInfoList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetSize(self):
        ...

    def GetMemoryRegionContainingAddress(self, addr, region_info):
        ...

    def GetMemoryRegionAtIndex(self, idx, region_info):
        ...

    def Append(self, *args):
        ...

    def Clear(self):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...


class SBModule:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def IsFileBacked(self):
        ...

    def GetFileSpec(self):
        ...

    def GetPlatformFileSpec(self):
        ...

    def SetPlatformFileSpec(self, platform_file):
        ...

    def GetRemoteInstallFileSpec(self):
        ...

    def SetRemoteInstallFileSpec(self, file):
        ...

    def GetByteOrder(self):
        ...

    def GetAddressByteSize(self):
        ...

    def GetTriple(self):
        ...

    def GetUUIDBytes(self):
        ...

    def GetUUIDString(self):
        ...

    def FindSection(self, sect_name):
        ...

    def ResolveFileAddress(self, vm_addr):
        ...

    def ResolveSymbolContextForAddress(self, addr, resolve_scope):
        ...

    def GetDescription(self, description):
        ...

    def GetNumCompileUnits(self):
        ...

    def GetCompileUnitAtIndex(self, arg2):
        ...

    def FindCompileUnits(self, sb_file_spec):
        ...

    def GetNumSymbols(self):
        ...

    def GetSymbolAtIndex(self, idx):
        ...

    def FindSymbol(self, *args):
        ...

    def FindSymbols(self, *args):
        ...

    def GetNumSections(self):
        ...

    def GetSectionAtIndex(self, idx):
        ...

    def FindFunctions(self, *args):
        ...

    def FindGlobalVariables(self, target, name, max_matches):
        ...

    def FindFirstGlobalVariable(self, target, name):
        ...

    def FindFirstType(self, name):
        ...

    def FindTypes(self, type):
        ...

    def GetTypeByID(self, uid):
        ...

    def GetBasicType(self, type):
        ...

    def GetTypes(self, *args):
        ...

    def GetVersion(self):
        ...

    def GetSymbolFileSpec(self):
        ...

    def GetObjectFileHeaderAddress(self):
        ...

    def GetObjectFileEntryPointAddress(self):
        ...

    @staticmethod
    def GetNumberAllocatedModules():
        ...

    @staticmethod
    def GarbageCollectAllocatedModules():
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def section_iter(self):
        ...

    def compile_unit_iter(self):
        ...

    def symbol_in_section_iter(
        self, section
    ) -> Generator[Incomplete, None, None]:
        ...

    class symbols_access:
        re_compile_type: Incomplete
        sbmodule: Incomplete

        def __init__(self, sbmodule) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_symbols_access_object(self):
        ...

    def get_compile_units_access_object(self):
        ...

    def get_symbols_array(self):
        ...

    class sections_access:
        re_compile_type: Incomplete
        sbmodule: Incomplete

        def __init__(self, sbmodule) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    class compile_units_access:
        re_compile_type: Incomplete
        sbmodule: Incomplete

        def __init__(self, sbmodule) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_sections_access_object(self):
        ...

    sections_array: Incomplete

    def get_sections_array(self):
        ...

    compile_units_array: Incomplete

    def get_compile_units_array(self):
        ...

    symbols: Incomplete
    symbol: Incomplete
    sections: Incomplete
    compile_units: Incomplete
    section: Incomplete

    def get_uuid(self):
        ...

    uuid: Incomplete
    file: Incomplete
    platform_file: Incomplete
    byte_order: Incomplete
    addr_size: Incomplete
    triple: Incomplete
    num_symbols: Incomplete
    num_sections: Incomplete


class SBModuleSpec:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetFileSpec(self):
        ...

    def SetFileSpec(self, fspec):
        ...

    def GetPlatformFileSpec(self):
        ...

    def SetPlatformFileSpec(self, fspec):
        ...

    def GetSymbolFileSpec(self):
        ...

    def SetSymbolFileSpec(self, fspec):
        ...

    def GetObjectName(self):
        ...

    def SetObjectName(self, name):
        ...

    def GetTriple(self):
        ...

    def SetTriple(self, triple):
        ...

    def GetUUIDBytes(self):
        ...

    def GetUUIDLength(self):
        ...

    def SetUUIDBytes(self, uuid, uuid_len):
        ...

    def GetObjectOffset(self):
        ...

    def SetObjectOffset(self, object_offset):
        ...

    def GetObjectSize(self):
        ...

    def SetObjectSize(self, object_size):
        ...

    def GetDescription(self, description):
        ...


class SBModuleSpecList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    @staticmethod
    def GetModuleSpecifications(path):
        ...

    def Append(self, *args):
        ...

    def FindFirstMatchingSpec(self, match_spec):
        ...

    def FindMatchingSpecs(self, match_spec):
        ...

    def GetSize(self):
        ...

    def GetSpecAtIndex(self, i):
        ...

    def GetDescription(self, description):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...


class SBPlatformConnectOptions:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetURL(self):
        ...

    def SetURL(self, url):
        ...

    def GetRsyncEnabled(self):
        ...

    def EnableRsync(self, options, remote_path_prefix, omit_remote_hostname):
        ...

    def DisableRsync(self):
        ...

    def GetLocalCacheDirectory(self):
        ...

    def SetLocalCacheDirectory(self, path):
        ...


class SBPlatformShellCommand:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def Clear(self):
        ...

    def GetShell(self):
        ...

    def SetShell(self, shell):
        ...

    def GetCommand(self):
        ...

    def SetCommand(self, shell_command):
        ...

    def GetWorkingDirectory(self):
        ...

    def SetWorkingDirectory(self, path):
        ...

    def GetTimeoutSeconds(self):
        ...

    def SetTimeoutSeconds(self, sec):
        ...

    def GetSignal(self):
        ...

    def GetStatus(self):
        ...

    def GetOutput(self):
        ...


class SBPlatform:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    @staticmethod
    def GetHostPlatform():
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetWorkingDirectory(self):
        ...

    def SetWorkingDirectory(self, path):
        ...

    def GetName(self):
        ...

    def ConnectRemote(self, connect_options):
        ...

    def DisconnectRemote(self):
        ...

    def IsConnected(self):
        ...

    def GetTriple(self):
        ...

    def GetHostname(self):
        ...

    def GetOSBuild(self):
        ...

    def GetOSDescription(self):
        ...

    def GetOSMajorVersion(self):
        ...

    def GetOSMinorVersion(self):
        ...

    def GetOSUpdateVersion(self):
        ...

    def SetSDKRoot(self, sysroot):
        ...

    def Put(self, src, dst):
        ...

    def Get(self, src, dst):
        ...

    def Install(self, src, dst):
        ...

    def Run(self, shell_command):
        ...

    def Launch(self, launch_info):
        ...

    def Attach(self, attach_info, debugger, target, error):
        ...

    def GetAllProcesses(self, error):
        ...

    def Kill(self, pid):
        ...

    def MakeDirectory(self, *args):
        ...

    def GetFilePermissions(self, path):
        ...

    def SetFilePermissions(self, path, file_permissions):
        ...

    def GetUnixSignals(self):
        ...

    def GetEnvironment(self):
        ...

    def SetLocateModuleCallback(self, callback):
        ...


class SBProcess:
    thisown: Incomplete
    eBroadcastBitStateChanged: Incomplete
    eBroadcastBitInterrupt: Incomplete
    eBroadcastBitSTDOUT: Incomplete
    eBroadcastBitSTDERR: Incomplete
    eBroadcastBitProfileData: Incomplete
    eBroadcastBitStructuredData: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    @staticmethod
    def GetBroadcasterClassName():
        ...

    def GetPluginName(self):
        ...

    def GetShortPluginName(self):
        ...

    def Clear(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetTarget(self):
        ...

    def GetByteOrder(self):
        ...

    def PutSTDIN(self, src):
        ...

    def GetSTDOUT(self, dst):
        ...

    def GetSTDERR(self, dst):
        ...

    def GetAsyncProfileData(self, dst):
        ...

    def ReportEventState(self, *args):
        ...

    def AppendEventStateReport(self, event, result):
        ...

    def RemoteAttachToProcessWithID(self, pid, error):
        ...

    def RemoteLaunch(
        self,
        argv,
        envp,
        stdin_path,
        stdout_path,
        stderr_path,
        working_directory,
        launch_flags,
        stop_at_entry,
        error,
    ):
        ...

    def GetNumThreads(self):
        ...

    def GetThreadAtIndex(self, index):
        ...

    def GetThreadByID(self, sb_thread_id):
        ...

    def GetThreadByIndexID(self, index_id):
        ...

    def GetSelectedThread(self) -> "SBThread":
        ...

    def CreateOSPluginThread(self, tid, context):
        ...

    def SetSelectedThread(self, thread):
        ...

    def SetSelectedThreadByID(self, tid):
        ...

    def SetSelectedThreadByIndexID(self, index_id):
        ...

    def GetNumQueues(self):
        ...

    def GetQueueAtIndex(self, index):
        ...

    def GetState(self):
        ...

    def GetExitStatus(self):
        ...

    def GetExitDescription(self):
        ...

    def GetProcessID(self):
        ...

    def GetUniqueID(self):
        ...

    def GetAddressByteSize(self):
        ...

    def Destroy(self):
        ...

    def Continue(self):
        ...

    def Stop(self):
        ...

    def Kill(self):
        ...

    def Detach(self, *args):
        ...

    def Signal(self, signal):
        ...

    def GetUnixSignals(self):
        ...

    def SendAsyncInterrupt(self):
        ...

    def GetStopID(self, include_expression_stops: bool = False):
        ...

    def GetStopEventForStopID(self, stop_id):
        ...

    def ForceScriptedState(self, new_state):
        ...

    def ReadMemory(self, addr, buf, error):
        ...

    def WriteMemory(self, addr, buf, error):
        ...

    def ReadCStringFromMemory(self, addr, char_buf, error):
        ...

    def ReadUnsignedFromMemory(self, addr, byte_size, error):
        ...

    def ReadPointerFromMemory(self, addr, error):
        ...

    @staticmethod
    def GetStateFromEvent(event):
        ...

    @staticmethod
    def GetRestartedFromEvent(event):
        ...

    @staticmethod
    def GetNumRestartedReasonsFromEvent(event):
        ...

    @staticmethod
    def GetRestartedReasonAtIndexFromEvent(event, idx):
        ...

    @staticmethod
    def GetProcessFromEvent(event):
        ...

    @staticmethod
    def GetInterruptedFromEvent(event):
        ...

    @staticmethod
    def GetStructuredDataFromEvent(event):
        ...

    @staticmethod
    def EventIsProcessEvent(event):
        ...

    @staticmethod
    def EventIsStructuredDataEvent(event):
        ...

    def GetBroadcaster(self):
        ...

    @staticmethod
    def GetBroadcasterClass():
        ...

    def GetDescription(self, description):
        ...

    def GetExtendedCrashInformation(self):
        ...

    def GetNumSupportedHardwareWatchpoints(self, error):
        ...

    def LoadImage(self, *args):
        ...

    def LoadImageUsingPaths(self, image_spec, paths, loaded_path, error):
        ...

    def UnloadImage(self, image_token):
        ...

    def SendEventData(self, data):
        ...

    def GetNumExtendedBacktraceTypes(self):
        ...

    def GetExtendedBacktraceTypeAtIndex(self, idx):
        ...

    def GetHistoryThreads(self, addr):
        ...

    def IsInstrumentationRuntimePresent(self, type):
        ...

    def SaveCore(self, *args):
        ...

    def GetMemoryRegionInfo(self, load_addr, region_info):
        ...

    def GetMemoryRegions(self):
        ...

    def GetProcessInfo(self):
        ...

    def AllocateMemory(self, size, permissions, error):
        ...

    def DeallocateMemory(self, ptr):
        ...

    def GetScriptedImplementation(self):
        ...

    def WriteMemoryAsCString(self, addr, str, error):
        ...

    def __get_is_alive__(self):
        ...

    def __get_is_running__(self):
        ...

    def __get_is_stopped__(self):
        ...

    class threads_access:
        sbprocess: Incomplete

        def __init__(self, sbprocess) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_threads_access_object(self):
        ...

    def get_process_thread_list(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def __int__(self) -> int:
        ...

    threads: Incomplete
    thread: Incomplete
    is_alive: Incomplete
    is_running: Incomplete
    is_stopped: Incomplete
    id: Incomplete
    target: Incomplete
    num_threads: Incomplete
    selected_thread: Incomplete
    state: Incomplete
    exit_state: Incomplete
    exit_description: Incomplete
    broadcaster: Incomplete


class SBProcessInfo:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetName(self):
        ...

    def GetExecutableFile(self):
        ...

    def GetProcessID(self):
        ...

    def GetUserID(self):
        ...

    def GetGroupID(self):
        ...

    def UserIDIsValid(self):
        ...

    def GroupIDIsValid(self):
        ...

    def GetEffectiveUserID(self):
        ...

    def GetEffectiveGroupID(self):
        ...

    def EffectiveUserIDIsValid(self):
        ...

    def EffectiveGroupIDIsValid(self):
        ...

    def GetParentProcessID(self):
        ...

    def GetTriple(self):
        ...


class SBProcessInfoList:
    thisown: Incomplete
    __swig_destroy__: Incomplete

    def __init__(self, *args) -> None:
        ...

    def GetSize(self):
        ...

    def GetProcessInfoAtIndex(self, idx, info):
        ...

    def Clear(self):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...


class SBQueue:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetProcess(self):
        ...

    def GetQueueID(self):
        ...

    def GetName(self):
        ...

    def GetIndexID(self):
        ...

    def GetNumThreads(self):
        ...

    def GetThreadAtIndex(self, arg2):
        ...

    def GetNumPendingItems(self):
        ...

    def GetPendingItemAtIndex(self, arg2):
        ...

    def GetNumRunningItems(self):
        ...

    def GetKind(self):
        ...


class SBQueueItem:
    thisown: Incomplete

    def __init__(self) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetKind(self):
        ...

    def SetKind(self, kind):
        ...

    def GetAddress(self):
        ...

    def SetAddress(self, addr):
        ...

    def GetExtendedBacktraceThread(self, type):
        ...

    def __hex__(self):
        ...


class SBReproducer:
    thisown: Incomplete

    @staticmethod
    def Capture(path):
        ...

    @staticmethod
    def PassiveReplay(path):
        ...

    @staticmethod
    def SetAutoGenerate(b):
        ...

    @staticmethod
    def SetWorkingDirectory(path):
        ...

    def __init__(self) -> None:
        ...

    __swig_destroy__: Incomplete


class SBScriptObject:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetPointer(self):
        ...

    def GetLanguage(self):
        ...

    ptr: Incomplete
    lang: Incomplete


class SBSection:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetName(self):
        ...

    def GetParent(self):
        ...

    def FindSubSection(self, sect_name):
        ...

    def GetNumSubSections(self):
        ...

    def GetSubSectionAtIndex(self, idx):
        ...

    def GetFileAddress(self):
        ...

    def GetLoadAddress(self, target):
        ...

    def GetByteSize(self):
        ...

    def GetFileOffset(self):
        ...

    def GetFileByteSize(self):
        ...

    def GetSectionData(self, *args):
        ...

    def GetSectionType(self):
        ...

    def GetPermissions(self):
        ...

    def GetTargetByteSize(self):
        ...

    def GetAlignment(self):
        ...

    def GetDescription(self, description):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def get_addr(self):
        ...

    name: Incomplete
    addr: Incomplete
    file_addr: Incomplete
    size: Incomplete
    file_offset: Incomplete
    file_size: Incomplete
    data: Incomplete
    type: Incomplete
    target_byte_size: Incomplete
    alignment: Incomplete


class SBSourceManager:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def DisplaySourceLinesWithLineNumbers(
        self, file, line, context_before, context_after, current_line_cstr, s
    ):
        ...

    def DisplaySourceLinesWithLineNumbersAndColumn(
        self,
        file,
        line,
        column,
        context_before,
        context_after,
        current_line_cstr,
        s,
    ):
        ...


class SBStream:
    thisown: Incomplete

    def __init__(self) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetData(self):
        ...

    def GetSize(self):
        ...

    def Print(self, str):
        ...

    def RedirectToFile(self, *args):
        ...

    def RedirectToFileDescriptor(self, fd, transfer_fh_ownership):
        ...

    def Clear(self):
        ...

    def __len__(self) -> int:
        ...

    def RedirectToFileHandle(self, file, transfer_fh_ownership):
        ...

    def write(self, str):
        ...

    def flush(self):
        ...


class SBStringList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def AppendString(self, str):
        ...

    def AppendList(self, *args):
        ...

    def GetSize(self):
        ...

    def GetStringAtIndex(self, *args):
        ...

    def Clear(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...


class SBStructuredData:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def SetFromJSON(self, *args: Any) -> None:
        ...

    def Clear(self):
        ...

    def GetAsJSON(self, stream: Any) -> str:
        ...

    def GetDescription(self, stream):
        ...

    def GetType(self):
        ...

    def GetSize(self):
        ...

    def GetKeys(self, keys):
        ...

    def GetValueForKey(self, key):
        ...

    def GetItemAtIndex(self, idx):
        ...

    def GetUnsignedIntegerValue(self, fail_value: int = 0):
        ...

    def GetSignedIntegerValue(self, fail_value: int = 0):
        ...

    def GetIntegerValue(self, fail_value: int = 0):
        ...

    def GetFloatValue(self, fail_value: float = 0.0):
        ...

    def GetBooleanValue(self, fail_value: bool = False):
        ...

    def GetStringValue(self, dst):
        ...

    def GetGenericValue(self):
        ...

    def __int__(self) -> int:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...


class SBSymbol:
    thisown: Incomplete
    __swig_destroy__: Incomplete

    def __init__(self, *args) -> None:
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetName(self):
        ...

    def GetDisplayName(self):
        ...

    def GetMangledName(self):
        ...

    def GetInstructions(self, *args):
        ...

    def GetStartAddress(self):
        ...

    def GetEndAddress(self):
        ...

    def GetValue(self):
        ...

    def GetSize(self):
        ...

    def GetPrologueByteSize(self):
        ...

    def GetType(self):
        ...

    def GetDescription(self, description):
        ...

    def IsExternal(self):
        ...

    def IsSynthetic(self):
        ...

    def __hex__(self):
        ...

    def get_instructions_from_current_target(self):
        ...

    name: Incomplete
    mangled: Incomplete
    type: Incomplete
    addr: Incomplete
    end_addr: Incomplete
    prologue_size: Incomplete
    instructions: Incomplete
    external: Incomplete
    synthetic: Incomplete


class SBSymbolContext:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetModule(self):
        ...

    def GetCompileUnit(self):
        ...

    def GetFunction(self):
        ...

    def GetBlock(self):
        ...

    def GetLineEntry(self):
        ...

    def GetSymbol(self):
        ...

    def SetModule(self, module):
        ...

    def SetCompileUnit(self, compile_unit):
        ...

    def SetFunction(self, function):
        ...

    def SetBlock(self, block):
        ...

    def SetLineEntry(self, line_entry):
        ...

    def SetSymbol(self, symbol):
        ...

    def GetParentOfInlinedScope(self, curr_frame_pc, parent_frame_addr):
        ...

    def GetDescription(self, description):
        ...

    module: Incomplete
    compile_unit: Incomplete
    function: Incomplete
    block: Incomplete
    symbol: Incomplete
    line_entry: Incomplete


class SBSymbolContextList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetSize(self):
        ...

    def GetContextAtIndex(self, idx):
        ...

    def GetDescription(self, description):
        ...

    def Append(self, *args):
        ...

    def Clear(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, key):
        ...

    def get_module_array(self):
        ...

    def get_compile_unit_array(self):
        ...

    def get_function_array(self):
        ...

    def get_block_array(self):
        ...

    def get_symbol_array(self):
        ...

    def get_line_entry_array(self):
        ...

    modules: Incomplete
    compile_units: Incomplete
    functions: Incomplete
    blocks: Incomplete
    line_entries: Incomplete
    symbols: Incomplete


class SBTarget:
    thisown: Incomplete
    eBroadcastBitBreakpointChanged: Incomplete
    eBroadcastBitModulesLoaded: Incomplete
    eBroadcastBitModulesUnloaded: Incomplete
    eBroadcastBitWatchpointChanged: Incomplete
    eBroadcastBitSymbolsLoaded: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    @staticmethod
    def EventIsTargetEvent(event):
        ...

    @staticmethod
    def GetTargetFromEvent(event):
        ...

    @staticmethod
    def GetNumModulesFromEvent(event):
        ...

    @staticmethod
    def GetModuleAtIndexFromEvent(idx, event):
        ...

    @staticmethod
    def GetBroadcasterClassName():
        ...

    def GetProcess(self):
        ...

    def SetCollectingStats(self, v):
        ...

    def GetCollectingStats(self):
        ...

    def GetStatistics(self):
        ...

    def GetPlatform(self):
        ...

    def GetEnvironment(self):
        ...

    def Install(self):
        ...

    def LoadCore(self, *args):
        ...

    def LaunchSimple(self, argv, envp, working_directory):
        ...

    def Launch(self, *args):
        ...

    def Attach(self, attach_info, error):
        ...

    def AttachToProcessWithID(self, listener, pid, error):
        ...

    def AttachToProcessWithName(self, listener, name, wait_for, error):
        ...

    def ConnectRemote(self, listener, url, plugin_name, error):
        ...

    def GetExecutable(self):
        ...

    def AppendImageSearchPath(self, _from, to, error):
        ...

    def AddModule(self, *args):
        ...

    def GetNumModules(self):
        ...

    def GetModuleAtIndex(self, idx):
        ...

    def RemoveModule(self, module):
        ...

    def GetDebugger(self):
        ...

    def FindModule(self, file_spec):
        ...

    def FindCompileUnits(self, sb_file_spec):
        ...

    def GetByteOrder(self):
        ...

    def GetAddressByteSize(self):
        ...

    def GetTriple(self):
        ...

    def GetABIName(self):
        ...

    def GetLabel(self):
        ...

    def SetLabel(self, label):
        ...

    def GetDataByteSize(self):
        ...

    def GetCodeByteSize(self):
        ...

    def GetMaximumNumberOfChildrenToDisplay(self):
        ...

    def SetSectionLoadAddress(self, section, section_base_addr):
        ...

    def ClearSectionLoadAddress(self, section):
        ...

    def SetModuleLoadAddress(self, module, sections_offset):
        ...

    def ClearModuleLoadAddress(self, module):
        ...

    def FindFunctions(self, *args):
        ...

    def FindFirstGlobalVariable(self, name):
        ...

    def FindGlobalVariables(self, *args):
        ...

    def FindGlobalFunctions(self, name, max_matches, matchtype):
        ...

    def Clear(self):
        ...

    def ResolveFileAddress(self, file_addr):
        ...

    def ResolveLoadAddress(self, vm_addr):
        ...

    def ResolvePastLoadAddress(self, stop_id, vm_addr):
        ...

    def ResolveSymbolContextForAddress(self, addr, resolve_scope):
        ...

    def ReadMemory(self, addr, buf, error):
        ...

    def BreakpointCreateByLocation(self, *args: Any) -> SBBreakpoint:
        ...

    def BreakpointCreateByName(self, *args):
        ...

    def BreakpointCreateByNames(self, *args):
        ...

    def BreakpointCreateByRegex(self, *args):
        ...

    def BreakpointCreateBySourceRegex(self, *args):
        ...

    def BreakpointCreateForException(self, language, catch_bp, throw_bp):
        ...

    def BreakpointCreateByAddress(self, address):
        ...

    def BreakpointCreateBySBAddress(self, address):
        ...

    def BreakpointCreateFromScript(
        self,
        class_name,
        extra_args,
        module_list,
        file_list,
        request_hardware: bool = False,
    ):
        ...

    def BreakpointsCreateFromFile(self, *args):
        ...

    def BreakpointsWriteToFile(self, *args):
        ...

    def GetNumBreakpoints(self):
        ...

    def GetBreakpointAtIndex(self, idx):
        ...

    def BreakpointDelete(self, break_id):
        ...

    def FindBreakpointByID(self, break_id):
        ...

    def FindBreakpointsByName(self, name, bkpt_list):
        ...

    def GetBreakpointNames(self, names):
        ...

    def DeleteBreakpointName(self, name):
        ...

    def EnableAllBreakpoints(self):
        ...

    def DisableAllBreakpoints(self):
        ...

    def DeleteAllBreakpoints(self):
        ...

    def GetNumWatchpoints(self):
        ...

    def GetWatchpointAtIndex(self, idx):
        ...

    def DeleteWatchpoint(self, watch_id):
        ...

    def FindWatchpointByID(self, watch_id):
        ...

    def WatchAddress(self, addr, size, read, modify, error):
        ...

    def WatchpointCreateByAddress(self, addr, size, options, error):
        ...

    def EnableAllWatchpoints(self):
        ...

    def DisableAllWatchpoints(self):
        ...

    def DeleteAllWatchpoints(self):
        ...

    def GetBroadcaster(self):
        ...

    def FindFirstType(self, type):
        ...

    def FindTypes(self, type):
        ...

    def GetBasicType(self, type):
        ...

    def CreateValueFromAddress(self, name, addr, type):
        ...

    def CreateValueFromData(self, name, data, type):
        ...

    def CreateValueFromExpression(self, name, expr):
        ...

    def GetSourceManager(self):
        ...

    def ReadInstructions(self, *args):
        ...

    def GetInstructions(self, base_addr, buf):
        ...

    def GetInstructionsWithFlavor(self, base_addr, flavor_string, buf):
        ...

    def FindSymbols(self, *args):
        ...

    def GetDescription(self, description, description_level):
        ...

    def EvaluateExpression(self, *args):
        ...

    def GetStackRedZoneSize(self):
        ...

    def IsLoaded(self, module):
        ...

    def GetLaunchInfo(self):
        ...

    def SetLaunchInfo(self, launch_info):
        ...

    def GetTrace(self):
        ...

    def CreateTrace(self, error):
        ...

    class modules_access:
        sbtarget: Incomplete

        def __init__(self, sbtarget) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_modules_access_object(self):
        ...

    def get_modules_array(self):
        ...

    def module_iter(self):
        ...

    def breakpoint_iter(self):
        ...

    class bkpts_access:
        sbtarget: Incomplete

        def __init__(self, sbtarget) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_bkpts_access_object(self):
        ...

    def get_target_bkpts(self):
        ...

    def watchpoint_iter(self):
        ...

    class watchpoints_access:
        sbtarget: Incomplete

        def __init__(self, sbtarget) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_watchpoints_access_object(self):
        ...

    def get_target_watchpoints(self):
        ...

    modules: Incomplete
    module: Incomplete
    process: Incomplete
    executable: Incomplete
    debugger: Incomplete
    num_breakpoints: Incomplete
    breakpoints: Incomplete
    breakpoint: Incomplete
    num_watchpoints: Incomplete
    watchpoints: Incomplete
    watchpoint: Incomplete
    broadcaster: Incomplete
    byte_order: Incomplete
    addr_size: Incomplete
    triple: Incomplete
    data_byte_size: Incomplete
    code_byte_size: Incomplete
    platform: Incomplete


class SBThread:
    thisown: Incomplete
    eBroadcastBitStackChanged: Incomplete
    eBroadcastBitThreadSuspended: Incomplete
    eBroadcastBitThreadResumed: Incomplete
    eBroadcastBitSelectedFrameChanged: Incomplete
    eBroadcastBitThreadSelected: Incomplete

    @staticmethod
    def GetBroadcasterClassName():
        ...

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def GetQueue(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetStopReason(self):
        ...

    def GetStopReasonDataCount(self):
        ...

    def GetStopReasonDataAtIndex(self, idx):
        ...

    def GetStopReasonExtendedInfoAsJSON(self, stream):
        ...

    def GetStopReasonExtendedBacktraces(self, type):
        ...

    def GetStopDescription(self, dst_or_null):
        ...

    def GetStopReturnValue(self):
        ...

    def GetThreadID(self):
        ...

    def GetIndexID(self):
        ...

    def GetName(self):
        ...

    def GetQueueName(self):
        ...

    def GetQueueID(self):
        ...

    def GetInfoItemByPathAsString(self, path, strm):
        ...

    def StepOver(self, *args: Any):
        ...

    def StepInto(self, *args: Any):
        ...

    def StepOut(self, *args: Any):
        ...

    def StepOutOfFrame(self, *args):
        ...

    def StepInstruction(self, *args):
        ...

    def StepOverUntil(self, frame, file_spec, line):
        ...

    def StepUsingScriptedThreadPlan(self, *args):
        ...

    def JumpToLine(self, file_spec, line):
        ...

    def RunToAddress(self, *args):
        ...

    def ReturnFromFrame(self, frame, return_value):
        ...

    def UnwindInnermostExpression(self):
        ...

    def Suspend(self, *args):
        ...

    def Resume(self, *args):
        ...

    def IsSuspended(self):
        ...

    def IsStopped(self):
        ...

    def GetNumFrames(self):
        ...

    def GetFrameAtIndex(self, idx: int) -> SBFrame:
        ...

    def GetSelectedFrame(self):
        ...

    def SetSelectedFrame(self, frame_idx):
        ...

    @staticmethod
    def EventIsThreadEvent(event):
        ...

    @staticmethod
    def GetStackFrameFromEvent(event):
        ...

    @staticmethod
    def GetThreadFromEvent(event):
        ...

    def GetProcess(self):
        ...

    def GetDescription(self, *args):
        ...

    def GetDescriptionWithFormat(self, format, output):
        ...

    def GetStatus(self, status):
        ...

    def GetExtendedBacktraceThread(self, type):
        ...

    def GetExtendedBacktraceOriginatingIndexID(self):
        ...

    def GetCurrentException(self):
        ...

    def GetCurrentExceptionBacktrace(self):
        ...

    def SafeToCallFunctions(self):
        ...

    def GetSiginfo(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    class frames_access:
        sbthread: Incomplete

        def __init__(self, sbthread) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_frames_access_object(self):
        ...

    def get_thread_frames(self):
        ...

    id: Incomplete
    idx: Incomplete
    return_value: Incomplete
    process: Incomplete
    num_frames: Incomplete
    frames: Incomplete
    frame: Incomplete
    name: Incomplete
    queue: Incomplete
    queue_id: Incomplete
    stop_reason: Incomplete
    is_suspended: Incomplete
    is_stopped: Incomplete


class SBThreadCollection:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetSize(self):
        ...

    def GetThreadAtIndex(self, idx):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...


class SBThreadPlan:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def Clear(self):
        ...

    def GetStopReason(self):
        ...

    def GetStopReasonDataCount(self):
        ...

    def GetStopReasonDataAtIndex(self, idx):
        ...

    def GetThread(self):
        ...

    def GetDescription(self, description):
        ...

    def SetPlanComplete(self, success):
        ...

    def IsPlanComplete(self):
        ...

    def IsPlanStale(self):
        ...

    def IsValid(self, *args):
        ...

    def GetStopOthers(self):
        ...

    def SetStopOthers(self, stop_others):
        ...

    def QueueThreadPlanForStepOverRange(self, *args):
        ...

    def QueueThreadPlanForStepInRange(self, *args):
        ...

    def QueueThreadPlanForStepOut(self, *args):
        ...

    def QueueThreadPlanForRunToAddress(self, *args):
        ...

    def QueueThreadPlanForStepScripted(self, *args):
        ...


class SBTrace:
    thisown: Incomplete

    def __init__(self) -> None:
        ...

    @staticmethod
    def LoadTraceFromFile(error, debugger, trace_description_file):
        ...

    def CreateNewCursor(self, error, thread):
        ...

    def SaveToDisk(self, error, bundle_dir, compact: bool = False):
        ...

    def GetStartConfigurationHelp(self):
        ...

    def Start(self, *args):
        ...

    def Stop(self, *args):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    __swig_destroy__: Incomplete


class SBTraceCursor:
    thisown: Incomplete

    def __init__(self) -> None:
        ...

    def SetForwards(self, forwards):
        ...

    def IsForwards(self):
        ...

    def Next(self):
        ...

    def HasValue(self):
        ...

    def GoToId(self, id):
        ...

    def HasId(self, id):
        ...

    def GetId(self):
        ...

    def Seek(self, offset, origin):
        ...

    def GetItemKind(self):
        ...

    def IsError(self):
        ...

    def GetError(self):
        ...

    def IsEvent(self):
        ...

    def GetEventType(self):
        ...

    def GetEventTypeAsString(self):
        ...

    def IsInstruction(self):
        ...

    def GetLoadAddress(self):
        ...

    def GetCPU(self):
        ...

    def IsValid(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__
    __swig_destroy__: Incomplete


class SBTypeMember:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetName(self):
        ...

    def GetType(self):
        ...

    def GetOffsetInBytes(self):
        ...

    def GetOffsetInBits(self):
        ...

    def IsBitfield(self):
        ...

    def GetBitfieldSizeInBits(self):
        ...

    def GetDescription(self, description, description_level):
        ...

    name: Incomplete
    type: Incomplete
    byte_offset: Incomplete
    bit_offset: Incomplete
    is_bitfield: Incomplete
    bitfield_bit_size: Incomplete


class SBTypeMemberFunction:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetName(self):
        ...

    def GetDemangledName(self):
        ...

    def GetMangledName(self):
        ...

    def GetType(self):
        ...

    def GetReturnType(self):
        ...

    def GetNumberOfArguments(self):
        ...

    def GetArgumentTypeAtIndex(self, arg2):
        ...

    def GetKind(self):
        ...

    def GetDescription(self, description, description_level):
        ...


class SBType:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetByteSize(self):
        ...

    def IsPointerType(self):
        ...

    def IsReferenceType(self):
        ...

    def IsFunctionType(self):
        ...

    def IsPolymorphicClass(self):
        ...

    def IsArrayType(self):
        ...

    def IsVectorType(self):
        ...

    def IsTypedefType(self):
        ...

    def IsAnonymousType(self):
        ...

    def IsScopedEnumerationType(self):
        ...

    def IsAggregateType(self):
        ...

    def GetPointerType(self):
        ...

    def GetPointeeType(self):
        ...

    def GetReferenceType(self):
        ...

    def GetTypedefedType(self):
        ...

    def GetDereferencedType(self):
        ...

    def GetUnqualifiedType(self):
        ...

    def GetArrayElementType(self):
        ...

    def GetArrayType(self, size):
        ...

    def GetVectorElementType(self):
        ...

    def GetCanonicalType(self):
        ...

    def GetEnumerationIntegerType(self):
        ...

    def GetBasicType(self, *args):
        ...

    def GetNumberOfFields(self):
        ...

    def GetNumberOfDirectBaseClasses(self):
        ...

    def GetNumberOfVirtualBaseClasses(self):
        ...

    def GetFieldAtIndex(self, idx):
        ...

    def GetDirectBaseClassAtIndex(self, idx):
        ...

    def GetVirtualBaseClassAtIndex(self, idx):
        ...

    def GetEnumMembers(self):
        ...

    def GetNumberOfTemplateArguments(self):
        ...

    def GetTemplateArgumentType(self, idx):
        ...

    def GetTemplateArgumentKind(self, idx):
        ...

    def GetFunctionReturnType(self):
        ...

    def GetFunctionArgumentTypes(self):
        ...

    def GetNumberOfMemberFunctions(self):
        ...

    def GetMemberFunctionAtIndex(self, idx):
        ...

    def GetModule(self):
        ...

    def GetName(self):
        ...

    def GetDisplayTypeName(self):
        ...

    def GetTypeClass(self):
        ...

    def IsTypeComplete(self):
        ...

    def GetTypeFlags(self):
        ...

    def GetDescription(self, description, description_level):
        ...

    def FindDirectNestedType(self, name):
        ...

    def template_arg_array(self):
        ...

    def __len__(self) -> int:
        ...

    module: Incomplete
    name: Incomplete
    size: Incomplete
    is_pointer: Incomplete
    is_reference: Incomplete
    num_fields: Incomplete
    num_bases: Incomplete
    num_vbases: Incomplete
    num_template_args: Incomplete
    template_args: Incomplete
    type: Incomplete
    is_complete: Incomplete

    def get_bases_array(self):
        ...

    def get_vbases_array(self):
        ...

    def get_fields_array(self):
        ...

    def get_members_array(self):
        ...

    def get_enum_members_array(self):
        ...

    bases: Incomplete
    vbases: Incomplete
    fields: Incomplete
    members: Incomplete
    enum_members: Incomplete


class SBTypeList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Append(self, type):
        ...

    def GetTypeAtIndex(self, index):
        ...

    def GetSize(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...


class SBTypeCategory:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetEnabled(self):
        ...

    def SetEnabled(self, arg2):
        ...

    def GetName(self):
        ...

    def GetLanguageAtIndex(self, idx):
        ...

    def GetNumLanguages(self):
        ...

    def AddLanguage(self, language):
        ...

    def GetDescription(self, description, description_level):
        ...

    def GetNumFormats(self):
        ...

    def GetNumSummaries(self):
        ...

    def GetNumFilters(self):
        ...

    def GetNumSynthetics(self):
        ...

    def GetTypeNameSpecifierForFilterAtIndex(self, arg2):
        ...

    def GetTypeNameSpecifierForFormatAtIndex(self, arg2):
        ...

    def GetTypeNameSpecifierForSummaryAtIndex(self, arg2):
        ...

    def GetTypeNameSpecifierForSyntheticAtIndex(self, arg2):
        ...

    def GetFilterForType(self, arg2):
        ...

    def GetFormatForType(self, arg2):
        ...

    def GetSummaryForType(self, arg2):
        ...

    def GetSyntheticForType(self, arg2):
        ...

    def GetFilterAtIndex(self, arg2):
        ...

    def GetFormatAtIndex(self, arg2):
        ...

    def GetSummaryAtIndex(self, arg2):
        ...

    def GetSyntheticAtIndex(self, arg2):
        ...

    def AddTypeFormat(self, arg2, arg3):
        ...

    def DeleteTypeFormat(self, arg2):
        ...

    def AddTypeSummary(self, arg2, arg3):
        ...

    def DeleteTypeSummary(self, arg2):
        ...

    def AddTypeFilter(self, arg2, arg3):
        ...

    def DeleteTypeFilter(self, arg2):
        ...

    def AddTypeSynthetic(self, arg2, arg3):
        ...

    def DeleteTypeSynthetic(self, arg2):
        ...

    class formatters_access_class:
        sbcategory: Incomplete
        get_count_function: Incomplete
        get_at_index_function: Incomplete
        get_by_name_function: Incomplete
        regex_type: Incomplete

        def __init__(
            self,
            sbcategory,
            get_count_function,
            get_at_index_function,
            get_by_name_function,
        ) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_formats_access_object(self):
        ...

    def get_formats_array(self):
        ...

    def get_summaries_access_object(self):
        ...

    def get_summaries_array(self):
        ...

    def get_synthetics_access_object(self):
        ...

    def get_synthetics_array(self):
        ...

    def get_filters_access_object(self):
        ...

    def get_filters_array(self):
        ...

    formats: Incomplete
    format: Incomplete
    summaries: Incomplete
    summary: Incomplete
    filters: Incomplete
    filter: Incomplete
    synthetics: Incomplete
    synthetic: Incomplete
    num_formats: Incomplete
    num_summaries: Incomplete
    num_filters: Incomplete
    num_synthetics: Incomplete
    name: Incomplete
    enabled: Incomplete


class SBTypeEnumMember:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetValueAsSigned(self):
        ...

    def GetValueAsUnsigned(self):
        ...

    def GetName(self):
        ...

    def GetType(self):
        ...

    def GetDescription(self, description, description_level):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    name: Incomplete
    type: Incomplete
    signed: Incomplete
    unsigned: Incomplete


class SBTypeEnumMemberList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Append(self, entry):
        ...

    def GetTypeEnumMemberAtIndex(self, index):
        ...

    def GetSize(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, key):
        ...


class SBTypeFilter:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetNumberOfExpressionPaths(self):
        ...

    def GetExpressionPathAtIndex(self, i):
        ...

    def ReplaceExpressionPathAtIndex(self, i, item):
        ...

    def AppendExpressionPath(self, item):
        ...

    def Clear(self):
        ...

    def GetOptions(self):
        ...

    def SetOptions(self, arg2):
        ...

    def GetDescription(self, description, description_level):
        ...

    def IsEqualTo(self, rhs):
        ...

    options: Incomplete
    count: Incomplete


class SBTypeFormat:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetFormat(self):
        ...

    def GetTypeName(self) -> str:
        ...

    def GetOptions(self):
        ...

    def SetFormat(self, arg2):
        ...

    def SetTypeName(self, arg2):
        ...

    def SetOptions(self, arg2):
        ...

    def GetDescription(self, description, description_level):
        ...

    def IsEqualTo(self, rhs):
        ...

    format: Incomplete
    options: Incomplete


class SBTypeNameSpecifier:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetName(self):
        ...

    def GetType(self):
        ...

    def GetMatchType(self):
        ...

    def IsRegex(self):
        ...

    def GetDescription(self, description, description_level):
        ...

    def IsEqualTo(self, rhs):
        ...

    name: Incomplete
    is_regex: Incomplete


class SBTypeSummaryOptions:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetLanguage(self):
        ...

    def GetCapping(self):
        ...

    def SetLanguage(self, arg2):
        ...

    def SetCapping(self, arg2):
        ...


class SBTypeSummary:
    thisown: Incomplete

    @staticmethod
    def CreateWithSummaryString(data, options: int = 0):
        ...

    @staticmethod
    def CreateWithFunctionName(data, options: int = 0):
        ...

    @staticmethod
    def CreateWithScriptCode(data, options: int = 0):
        ...

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def IsFunctionCode(self):
        ...

    def IsFunctionName(self):
        ...

    def IsSummaryString(self):
        ...

    def GetData(self):
        ...

    def SetSummaryString(self, data):
        ...

    def SetFunctionName(self, data):
        ...

    def SetFunctionCode(self, data):
        ...

    def GetOptions(self):
        ...

    def SetOptions(self, arg2):
        ...

    def GetDescription(self, description, description_level):
        ...

    def DoesPrintValue(self, value):
        ...

    def IsEqualTo(self, rhs):
        ...

    options: Incomplete
    is_summary_string: Incomplete
    is_function_name: Incomplete
    summary_data: Incomplete


class SBTypeSynthetic:
    thisown: Incomplete

    @staticmethod
    def CreateWithClassName(data, options: int = 0):
        ...

    @staticmethod
    def CreateWithScriptCode(data, options: int = 0):
        ...

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def IsClassCode(self):
        ...

    def IsClassName(self):
        ...

    def GetData(self):
        ...

    def SetClassName(self, data):
        ...

    def SetClassCode(self, data):
        ...

    def GetOptions(self):
        ...

    def SetOptions(self, arg2):
        ...

    def GetDescription(self, description, description_level):
        ...

    def IsEqualTo(self, rhs):
        ...

    options: Incomplete
    contains_code: Incomplete
    synthetic_data: Incomplete


class SBUnixSignals:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def Clear(self):
        ...

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetSignalAsCString(self, signo):
        ...

    def GetSignalNumberFromName(self, name):
        ...

    def GetShouldSuppress(self, signo):
        ...

    def SetShouldSuppress(self, signo, value):
        ...

    def GetShouldStop(self, signo):
        ...

    def SetShouldStop(self, signo, value):
        ...

    def GetShouldNotify(self, signo):
        ...

    def SetShouldNotify(self, signo, value):
        ...

    def GetNumSignals(self):
        ...

    def GetSignalAtIndex(self, index):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def get_unix_signals_list(self):
        ...

    threads: Incomplete


class SBValue:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def GetError(self) -> SBError:
        ...

    def GetID(self):
        ...

    def GetName(self):
        ...

    def GetTypeName(self) -> str:
        ...

    def GetDisplayTypeName(self) -> str:
        ...

    def GetByteSize(self):
        ...

    def IsInScope(self):
        ...

    def GetFormat(self):
        ...

    def SetFormat(self, format):
        ...

    def GetValue(self) -> str:
        ...

    def GetValueAsSigned(self, *args: Any) -> int:
        ...

    def GetValueAsUnsigned(self, *args: Any) -> int:
        ...

    def GetValueType(self):
        ...

    def GetValueDidChange(self):
        ...

    def GetSummary(self, *args: Any) -> str:
        ...

    def GetObjectDescription(self):
        ...

    def GetDynamicValue(self, use_dynamic):
        ...

    def GetStaticValue(self):
        ...

    def GetNonSyntheticValue(self):
        ...

    def GetPreferDynamicValue(self):
        ...

    def SetPreferDynamicValue(self, use_dynamic):
        ...

    def GetPreferSyntheticValue(self):
        ...

    def SetPreferSyntheticValue(self, use_synthetic):
        ...

    def IsDynamic(self):
        ...

    def IsSynthetic(self):
        ...

    def IsSyntheticChildrenGenerated(self):
        ...

    def SetSyntheticChildrenGenerated(self, arg2):
        ...

    def GetLocation(self):
        ...

    def SetValueFromCString(self, *args):
        ...

    def GetTypeFormat(self):
        ...

    def GetTypeSummary(self):
        ...

    def GetTypeFilter(self):
        ...

    def GetTypeSynthetic(self):
        ...

    def CreateChildAtOffset(self, name, offset, type):
        ...

    def Cast(self, type):
        ...

    def CreateValueFromExpression(self, *args):
        ...

    def CreateValueFromAddress(self, name, address, type):
        ...

    def CreateValueFromData(self, name, data, type):
        ...

    def GetChildAtIndex(self, *args: Any) -> "SBValue":
        ...

    def GetIndexOfChildWithName(self, name):
        ...

    def GetChildMemberWithName(self, *args: Any) -> "SBValue":
        ...

    def GetValueForExpressionPath(self, expr_path: str) -> "SBValue":
        ...

    def AddressOf(self):
        ...

    def GetLoadAddress(self):
        ...

    def GetAddress(self):
        ...

    def GetPointeeData(self, item_idx: int = 0, item_count: int = 1):
        ...

    def GetData(self):
        ...

    def SetData(self, data, error):
        ...

    def Clone(self, new_name):
        ...

    def GetDeclaration(self):
        ...

    def MightHaveChildren(self):
        ...

    def IsRuntimeSupportValue(self):
        ...

    def GetNumChildren(self, *args: Any) -> int:
        ...

    def GetOpaqueType(self):
        ...

    def GetTarget(self):
        ...

    def GetProcess(self):
        ...

    def GetThread(self):
        ...

    def GetFrame(self):
        ...

    def Dereference(self) -> "SBValue":
        ...

    def TypeIsPointerType(self):
        ...

    def GetType(self) -> SBType:
        ...

    def Persist(self):
        ...

    def GetDescription(self, description):
        ...

    def GetExpressionPath(self, *args):
        ...

    def EvaluateExpression(self, *args):
        ...

    def Watch(self, *args):
        ...

    def WatchPointee(self, resolve_location, read, write, error):
        ...

    def GetVTable(self):
        ...

    def __get_dynamic__(self):
        ...

    class children_access:
        sbvalue: Incomplete

        def __init__(self, sbvalue) -> None:
            ...

        def __len__(self) -> int:
            ...

        def __getitem__(self, key):
            ...

    def get_child_access_object(self):
        ...

    def get_value_child_list(self):
        ...

    def __hex__(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    children: Incomplete
    child: Incomplete
    name: Incomplete
    type: Incomplete
    size: Incomplete
    is_in_scope: Incomplete
    format: Incomplete
    value: Incomplete
    value_type: Incomplete
    changed: Incomplete
    data: Incomplete
    load_addr: Incomplete
    addr: Incomplete
    deref: Incomplete
    address_of: Incomplete
    error: Incomplete
    summary: Incomplete
    description: Incomplete
    dynamic: Incomplete
    location: Incomplete
    target: Incomplete
    process: Incomplete
    thread: Incomplete
    frame: Incomplete
    num_children: Incomplete
    unsigned: Incomplete
    signed: Incomplete

    def get_expr_path(self):
        ...

    path: Incomplete

    def synthetic_child_from_expression(
        self, name, expr, options: Incomplete = None
    ):
        ...

    def synthetic_child_from_data(self, name, data, type):
        ...

    def synthetic_child_from_address(self, name, addr, type):
        ...

    def linked_list_iter(
        self, next_item_name, end_of_list_test=...
    ) -> Generator[Incomplete, None, None]:
        ...


class SBValueList:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def Clear(self):
        ...

    def Append(self, *args):
        ...

    def GetSize(self):
        ...

    def GetValueAtIndex(self, idx):
        ...

    def GetFirstValueByName(self, name):
        ...

    def FindValueObjectByUID(self, uid):
        ...

    def GetError(self):
        ...

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, key):
        ...


class SBVariablesOptions:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetIncludeArguments(self):
        ...

    def SetIncludeArguments(self, arg2):
        ...

    def GetIncludeRecognizedArguments(self, arg2):
        ...

    def SetIncludeRecognizedArguments(self, arg2):
        ...

    def GetIncludeLocals(self):
        ...

    def SetIncludeLocals(self, arg2):
        ...

    def GetIncludeStatics(self):
        ...

    def SetIncludeStatics(self, arg2):
        ...

    def GetInScopeOnly(self):
        ...

    def SetInScopeOnly(self, arg2):
        ...

    def GetIncludeRuntimeSupportValues(self):
        ...

    def SetIncludeRuntimeSupportValues(self, arg2):
        ...

    def GetUseDynamic(self):
        ...

    def SetUseDynamic(self, arg2):
        ...


class SBWatchpoint:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def __nonzero__(self):
        ...

    __bool__ = __nonzero__

    def IsValid(self):
        ...

    def GetError(self):
        ...

    def GetID(self):
        ...

    def GetHardwareIndex(self):
        ...

    def GetWatchAddress(self):
        ...

    def GetWatchSize(self):
        ...

    def SetEnabled(self, enabled):
        ...

    def IsEnabled(self):
        ...

    def GetHitCount(self):
        ...

    def GetIgnoreCount(self):
        ...

    def SetIgnoreCount(self, n):
        ...

    def GetCondition(self):
        ...

    def SetCondition(self, condition):
        ...

    def GetDescription(self, description, level):
        ...

    def Clear(self):
        ...

    @staticmethod
    def EventIsWatchpointEvent(event):
        ...

    @staticmethod
    def GetWatchpointEventTypeFromEvent(event):
        ...

    @staticmethod
    def GetWatchpointFromEvent(event):
        ...

    def GetType(self):
        ...

    def GetWatchValueKind(self):
        ...

    def GetWatchSpec(self):
        ...

    def IsWatchingReads(self):
        ...

    def IsWatchingWrites(self):
        ...

    def __hex__(self):
        ...

    def __len__(self) -> int:
        ...


class SBWatchpointOptions:
    thisown: Incomplete

    def __init__(self, *args) -> None:
        ...

    __swig_destroy__: Incomplete

    def SetWatchpointTypeRead(self, read):
        ...

    def GetWatchpointTypeRead(self):
        ...

    def SetWatchpointTypeWrite(self, write_type):
        ...

    def GetWatchpointTypeWrite(self):
        ...


def in_range(symbol, section):
    ...


def command(command_name: Incomplete = None, doc: Incomplete = None):
    ...


class declaration:
    file: Incomplete
    line: Incomplete
    col: Incomplete

    def __init__(self, file, line, col) -> None:
        ...


class value_iter:
    def __iter__(self):
        ...

    def __next__(self):
        ...

    def next(self):
        ...

    def __len__(self) -> int:
        ...

    index: int
    length: int
    sbvalue: Incomplete

    def __init__(self, value) -> None:
        ...


class value:
    sbvalue: Incomplete

    def __init__(self, sbvalue) -> None:
        ...

    def __nonzero__(self):
        ...

    def __bool__(self) -> bool:
        ...

    def __getitem__(self, key):
        ...

    def __iter__(self):
        ...

    def __getattr__(self, name):
        ...

    def __add__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __floordiv__(self, other):
        ...

    def __mod__(self, other):
        ...

    def __divmod__(self, other):
        ...

    def __pow__(self, other):
        ...

    def __lshift__(self, other):
        ...

    def __rshift__(self, other):
        ...

    def __and__(self, other):
        ...

    def __xor__(self, other):
        ...

    def __or__(self, other):
        ...

    def __div__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __iadd__(self, other):
        ...

    def __isub__(self, other):
        ...

    def __imul__(self, other):
        ...

    def __idiv__(self, other):
        ...

    def __itruediv__(self, other):
        ...

    def __ifloordiv__(self, other):
        ...

    def __imod__(self, other):
        ...

    def __ipow__(self, other, modulo):
        ...

    def __ilshift__(self, other):
        ...

    def __irshift__(self, other):
        ...

    def __iand__(self, other):
        ...

    def __ixor__(self, other):
        ...

    def __ior__(self, other):
        ...

    def __neg__(self):
        ...

    def __pos__(self):
        ...

    def __abs__(self):
        ...

    def __invert__(self):
        ...

    def __complex__(self) -> complex:
        ...

    def __int__(self) -> int:
        ...

    def __long__(self):
        ...

    def __float__(self) -> float:
        ...

    def __oct__(self):
        ...

    def __hex__(self):
        ...

    def __len__(self) -> int:
        ...


class SBSyntheticValueProvider:
    def __init__(self, valobj) -> None:
        ...

    def num_children(self):
        ...

    def get_child_index(self, name) -> None:
        ...

    def get_child_at_index(self, idx) -> None:
        ...

    def update(self) -> None:
        ...

    def has_children(self):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...


def is_numeric_type(basic_type):
    ...


debugger_unique_id: int
debugger: Incomplete
target: Incomplete
process: Incomplete
thread: Incomplete
frame: Incomplete
