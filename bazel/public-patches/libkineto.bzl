"""Repository rule that fetches libkineto's sources from pytorch/pytorch."""

# BEGIN_GENERATED
# NOTE: Use 'update-libkineto' to update these values
LIBKINETO_COMMIT = "31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457"

LIBKINETO_SUBDIR = "third_party/kineto/libkineto"

LIBKINETO_FILES = {
    "include/AbstractConfig.h": "48c5ecd0b837b85a53c83065bac188fe3fb38056ce981c58f7343cc25f3e1981",
    "include/ActivityProfilerInterface.h": "21e480dc84e830762df8f6b70aa0866604306b7468fc4137084cb1cd21663040",
    "include/ActivityTraceInterface.h": "6db856fa1ce6f775b11136f6787ef6d993490a4f3df787421eefda3b60bf1975",
    "include/ActivityType.h": "e903237825a988e61756df0e76176113d695b60889cf56877d6c12371a25c9f0",
    "include/ClientInterface.h": "f9729ed3e3c55f2e9e6f0de4440aacef96fed0ee2d2f30c6afa7df18c98d535e",
    "include/Config.h": "72d1788714b6d0b73198e26f2ad3efa2d8ff2f661f12cfda69c9325496795cc2",
    "include/EnvMetadata.h": "817e2a0a63fc4d40f4dfb94977d2a3e44b589d5e8b8c3638c5992a45014d768c",
    "include/GenericTraceActivity.h": "e337011f9e95d3653d155953f863686879fb6441992d842b3dedc97d36aa0bc4",
    "include/IActivityProfiler.h": "90a09018c0dbb56a351e50013583ba248cae3c44d195401424d151b145e65067",
    "include/ILoggerObserver.h": "fcade4ef191e8f42e403a0d4f4d7bb917461f619268dffd8c730bfba34cd8196",
    "include/ITraceActivity.h": "33f147fb23719f479dffc6d59923bf8d9d6cc1b0c94e409d50c66fb1c966c997",
    "include/LoggingAPI.h": "1cb493e0caad91f3c2c8ac3bdf05f32dbae695e737d3ef3ac053eab43eb2b243",
    "include/MetadataFieldCatalog.h": "2211de18d0970a8563d2dc9227fd7f9cccd14008c575eb2d4def3dd3314b8d90",
    "include/ThreadUtil.h": "8a75ce8244dd4af1c7b7eca29e2b7d307f502f8aa4c21cba4e5793a19d46fb6a",
    "include/TraceSpan.h": "69311db5bb75d220e00383debf8e71c8063e1711dfb1995949259285d0944423",
    "include/TypedMetadata.h": "ae2856770fa64f099fe3627dc78f89bf1c83895e4a43ee739d3df9505ebb9483",
    "include/TypedMetadataJson.h": "256378b2787e0491ca80446f1c47c128bfce4523bba741a7af5721e4511de6a1",
    "include/libkineto.h": "b2cc69abaaad1909100349be7c0abe27c035d748856882bd2e9b694a61d5586e",
    "include/output_base.h": "67ad733e2a076a946ad88e1f2ea3f8e83dac60af7bae192992f7596419efdabf",
    "include/time_since_epoch.h": "9a3a6c35ab643ff99b979da672f1c94c845ded8e1069046a43788b6c0873f7f3",
    "libkineto_defs.bzl": "b12e88cb149cdbeea102c498fb5a22d0141d8f2f5e08f00c1b26ffcb30eddea6",
    "src/AbstractConfig.cpp": "005e088c11a6bea3076dd023d71fabd33276152b785c4531a30c5cf36c472027",
    "src/ActivityBuffers.h": "11f128cf1b255a8453e85225df42e8f9eee752079b6bcd50caf202b1627226c1",
    "src/ActivityLoggerFactory.h": "8ae0578c00347df4f2f738a15a9a7a8d3405d8cb8af3102ab16eaad4697dbb5b",
    "src/ActivityProfilerController.cpp": "294f8b88be6506049b8702c75c015fff0de7cf82695fb054ddf1e168493b7910",
    "src/ActivityProfilerController.h": "10842b19b26568eb2608fbecb268d53ed2a6dde5f5a037b28bf0941097d22fa0",
    "src/ActivityProfilerProxy.cpp": "fb9ba791af45786ea3b168c77311c5fe8a54bab0ab878a5c5199f7235ea48dc7",
    "src/ActivityProfilerProxy.h": "1125a64c528756c4572fc27b7165f670254b1d63c2ef808afb3d981c9cf3857c",
    "src/ActivityTrace.h": "86f1a6e7219c45b38b9c2af339fc9118ed121f9c83cad723afe09e761965bb4e",
    "src/ActivityType.cpp": "4aad9b5c20d6270fb82c5df33b49dcc731185d4a1e053f589bbdc39f41a7c883",
    "src/ApproximateClock.cpp": "322966cb1a1474fccff35616b89e4870df934d97aa034f4bf9d4c3c1ab99431d",
    "src/ApproximateClock.h": "2d6cc8c1b74aca6d5b7caaa73c3cb7a7871956b708516e45b72caf9811768f7f",
    "src/AsyncActivityProfilerHandler.cpp": "9320402631379fff8e5d264d8a9ff478d31f65127a1eab6300aec02c4ff1371b",
    "src/AsyncActivityProfilerHandler.h": "b88f3556bcdeb4f05907847866e0bf29bb6bc227aa8578e2e2eee1c3ea12eb9b",
    "src/Config.cpp": "e3a55cf6227c412204bbb04c780ad7fcfbec2101b541d07efc245074ad08b238",
    "src/ConfigLoader.cpp": "2537141718079a55604220ba7472e53ed237a447455648dbb2c063f34d43e12d",
    "src/ConfigLoader.h": "87438effd01060c9307c38d3111ddd80cd45dd044740c15632c4290142403aa6",
    "src/CuptiActivity.h": "35f858cc3a251eca168b19b3e9c248627aba1d7bf72d0ba4a70cfd44dab60af1",
    "src/CuptiActivityApi.cpp": "74807b6b8fd5dccd37d391c55a7dd44bbf950661c87c9fafa2cdc99f0470f74b",
    "src/CuptiActivityApi.h": "68ece5b56d9fae380a16f4eed843a27f1c7972652ba48138eaf8cbcf19f3589c",
    "src/CuptiActivityBuffer.h": "a3896ab7bb04d88848ed06231b4e5c7012c21247b808c711ab21374f2859831c",
    "src/CuptiActivityProfiler.cpp": "cffbaf0845cd496b73a042ba69681b7c6f699f17c7a1df5d0b07f0e03503bc43",
    "src/CuptiActivityProfiler.h": "26710fc328e4534639fb7bef54bc5de5d0eb1f4d1558f52cbf5fda942a1af92a",
    "src/CuptiCallbackApi.cpp": "a091b94615549910ec8e1f424078371128adcb28ec4067b1dcc3c1b8612fed54",
    "src/CuptiCallbackApi.h": "01fc027e875bb3793091219662829052d94ca3653b2475ae6f7b6172d883d734",
    "src/CuptiCallbackApiMock.h": "7e9fedb2c861e683628b7e2e5160e112dcc587e84443e171de0c467e28cf671a",
    "src/CuptiCbidRegistry.cpp": "0abcfa67bcfb53c3eea5462b4cb9d81967021870bf03f62f01f730ae53ac17a8",
    "src/CuptiCbidRegistry.h": "928da4773e781d98ab4e395fbcd0018729c788d3088fd8b860104b86b411ea77",
    "src/CuptiPMSamplingApi.cpp": "94f13c96e9bb96291298e45b79eb9ad84e639cce682ddfd325916438119e6071",
    "src/CuptiPMSamplingApi.h": "b7cd09b6b8a58c41f3cb090c10ef9df4ecab27e58ba0e266ce87c81c1f7dd959",
    "src/CuptiPMSamplingController.cpp": "f4d2b32734446f274b08c3ccebb7d5756d440736a872f047cd17cac08c9194ec",
    "src/CuptiPMSamplingController.h": "2ad0d5072c230afbd3918b7de0f57c0544f3bbafe7fff2223738fff171f6604f",
    "src/CuptiPMSamplingProfiler.cpp": "e2a35d182b36d9aea9e66e3be407f9241dbe33e0664d6aa7cb35755b9b9cac04",
    "src/CuptiPMSamplingProfiler.h": "572ebaf9f77fbe29bc92d969e12dfabad3a4bbff8898f4a33f7f31543173a25d",
    "src/CuptiTimestamp.cpp": "91566fa72ccb500874599d979cf65dfda66a3eb1e656137106fbfa451de6fd54",
    "src/CuptiTimestamp.h": "b3c1cd25ca14321bdc2e1649de7d4f7762ccb548b5d34e634b4d6a4325a340a1",
    "src/DaemonConfigLoader.cpp": "44ee5882ecfb70591e31873cf7a0931d67529f994c1f485304fe2ebdaf290815",
    "src/DaemonConfigLoader.h": "13ff1505851b85e453cc2a3942f6d4278f8dcde271f62b148775104e32f644e6",
    "src/Demangle.cpp": "705aeb16f6222d51daf10cc818ccaa2556906651cde8a9cb8c32aecdc1065876",
    "src/Demangle.h": "6abe0d9fc74bb34c1920b5968488a1ef3d2acc070fedab167713c733e5adfe8a",
    "src/DeviceProperties.cpp": "02cb3c291808b1407747a9e4fa2b9cc13c43c343484c3f277108a38feea3b34e",
    "src/DeviceProperties.h": "934fdbef4f4072652c45da4d67ce2b1281c474b6ef2aef268d9227caa743a36c",
    "src/DeviceUtil.cpp": "cd7079e73d8a1f5e98725073af26cc29fea908b919080c81e3283b7366ff254f",
    "src/DeviceUtil.h": "2cb439cba30a284546e04d3eeccb4f07b5f21916e9b740544d86a8738f670d4a",
    "src/GenericActivityProfiler.cpp": "e48dcf27de81445f0edf5a082a4dd4046512f23de0f65a33b5fc35a403c4aa4c",
    "src/GenericActivityProfiler.h": "a264df558b59c7325e4673fb3372d0db517750e1609fd69de6c87be099748057",
    "src/GenericTraceActivity.cpp": "aa049a83125797c830783b6f7503290946934e49f6222376e26b4ae520721358",
    "src/ILoggerObserver.cpp": "9c5dac611f7621398cc6e680584feeb097089187e02786cbf4924a2f900c1509",
    "src/InvariantViolations.h": "c9c06460189b7dd6de022da7ec03e404a1f5e193ec0779ab144db04d93eab19b",
    "src/IpcFabricConfigClient.cpp": "85175204b5db6da0d89c97886e48c61170eb7b290e5fbfde5b0201285d1484c7",
    "src/IpcFabricConfigClient.h": "6ece4b0cef5b32b4a8fb2f4774913f94456a02a22188710b0896cd4831c6a0da",
    "src/KernelRegistry.cpp": "0910d12c76d45b7600a8ddc12169e0924e6625c7db93c56423d6265d4faa9e01",
    "src/KernelRegistry.h": "25323202fc259a3164ee5f7ca5f70b0e2f56a1424bbeff8bc16b77f0badffd9f",
    "src/Logger.cpp": "b58af9595358e1468df29ef2633734f82b7617c78189e692ad297114e102176a",
    "src/Logger.h": "247112f5a84343f55a270da66e1e23b0e48ce484aa37f1c50dda620b48926037",
    "src/LoggerCollector.h": "9791dd24bf9f04d4e6e38cbad24155f0b3bee1c06b940b01c93f856877ab03fa",
    "src/LoggingAPI.cpp": "a6ea07dc1fa9e9cc4e1bf81df338097b40b66b93dc36e5f97d193931175bcba6",
    "src/RocLogger.cpp": "a7ca8119b54748e8a5bd14ddd51b43bb64eb788c53b9c84fd639c98d9ef8187a",
    "src/RocLogger.h": "98ab78a2e8cd5fe31e86a601afdb49717140c09025a596ffd86117748222122b",
    "src/RocmActivityProfiler.cpp": "eba4b63a5f05fe6b3c8fff129c87d2f29d3eabaa9b7e714a908d9d9629f7f65d",
    "src/RocmActivityProfiler.h": "3ac88e3441b30c66382f014b541d3965ecd5ab4d4daa744ec465ee0cf12bdcdd",
    "src/RocmStreamQueue.h": "4eaec9e91b31e8f2eef43f41817b96d2b336a04278cd6f94e7612707406926cc",
    "src/RocprofActivity.h": "4337dbe47eeb653c3902d30a68cf37687654d5582abcf25acd39eaa80b3a16a1",
    "src/RocprofActivityApi.cpp": "42a015f4449cc4dc73f368a1c03e4d766f1ecec48271dbbdbdd172e4379a1548",
    "src/RocprofActivityApi.h": "933f567af04b0367ec29bd359e798f5f9c77c99b633caf11e7b1207f415ec720",
    "src/RocprofActivity_inl.h": "5c73b6b7c9bdc5f4830c9c71abb5544cb17a999b5deb0de5eff1fa27d8ca87db",
    "src/RocprofLogger.cpp": "7380a5a7c09d49f9929326f39a2a8adaf102c4780daf7502fdbfa939b9328272",
    "src/RocprofLogger.h": "dd9d55dd7f3f010de5f8a88899e4edf31a1509ed9374e5788e48b79f0927b3da",
    "src/ScopeExit.h": "07426f9df5d7f9bbfd1a2ec822cbb22c7925254515f30a8f3aade84f3044629a",
    "src/SyncActivityProfilerHandler.cpp": "055d9f2e81ec43756c45b730f106e9451ca413ca4e50d2a9a9b76e1420d6d1d7",
    "src/SyncActivityProfilerHandler.h": "ee4dac540759c46fbd050487b4d25b161ed27edbe0cb8180741853090be53ab7",
    "src/ThreadUtil.cpp": "489a9e91785d690018ec138f65fc90a9161597caf76a6b6591965e2bbb2237d3",
    "src/ThrowUtil.h": "c153ba585ecd437e73be8b65a023d20632087bee135d8abfda394a25c4f5a4ac",
    "src/WeakSymbols.cpp": "ea808737772114ed0810ffcf6ecce1dd9df4fde9466d1d479a18acc6e549e997",
    "src/cupti_strings.cpp": "a9aaf0e9aa615ce9d85c955f2cf12871eaa0d5812950cf890ee26771a1fe0eac",
    "src/cupti_strings.h": "ce84066ae6878700409c58e38c86c0ad3140475edeed5eb73183268a91bbf83b",
    "src/init.cpp": "2ccd8bd84a9af74f8978f7363096752e444145869c95ac5fe3f0112126a80448",
    "src/libkineto_api.cpp": "f2c840ee34c09c9ed4de053b575257124d31f79f66b77c16eb577e89be861097",
    "src/output_json.cpp": "97460343a7538c8c302b03762d21ee69863d811b2a1459efb139909fde98b577",
    "src/output_json.h": "3e1cf9d54b483c48d7ca4d5e6d3f054abbaf25b00f890088ccd87793244ef1d8",
    "src/output_membuf.h": "b385e4911abc54a5fc72af5a1caeebf1e79a174ffdd327020f2b7192bfa84c37",
    "src/plugin/xpupti/CMakeLists.txt": "367f1f54b3885251bb118d098a6eb81d15bbbf82d096ca0f557b3b11f193fcae",
    "src/plugin/xpupti/FindSYCLToolkit.cmake": "bd86185bca359f57bc64a6fd8bc94ad71cc395865c1470594913481a7d96b28d",
    "src/plugin/xpupti/XpuptiActivityApi.cpp": "484c99d14ffef183f20dcec3ccc9a46af5a1881b3ce5b0e630ca9d36510cc448",
    "src/plugin/xpupti/XpuptiActivityApi.h": "055b7589785172c2cb202a19f1e3c41ece0ebd781701a6219448d23d9aa00197",
    "src/plugin/xpupti/XpuptiActivityBuffer.h": "39201a303fc07316cf5c4ba41a5a77b2c874ebffc13037479b0e84b3a0675594",
    "src/plugin/xpupti/XpuptiActivityHandlers.cpp": "247252f728e0dcc024416b36b690593d83f3d57ad05e05b1ffd68b2cea2fd92c",
    "src/plugin/xpupti/XpuptiActivityProfiler.cpp": "270ef30c28e0b3ef7758d51de15ab953609e4c564d1816e22b50737bd22ae92c",
    "src/plugin/xpupti/XpuptiActivityProfiler.h": "f30345336350ea216bc844d84e804d90c31c96d557419aaee05709514e3969f9",
    "src/plugin/xpupti/XpuptiActivityProfilerSession.cpp": "4d1c5f5dfa0498e92dd9c79c22c99091f9981518b1991e83ddbcb1de25829f12",
    "src/plugin/xpupti/XpuptiActivityProfilerSession.h": "a4b1391ed53bee4d7366aa8952fdb2295824e464c0046a1ad22cbb1169d6976a",
    "src/plugin/xpupti/XpuptiActivityTypeMask.h": "13135a1ed107b9db1879b3451fe59db0105319d0bafb6fc25b1ea9b43bd29bf4",
    "src/plugin/xpupti/XpuptiProfilerMacros.cpp": "a04b1bdf34175d9007ace2c0acbf47135b3c585a70cb4a4ef9dc1157689806db",
    "src/plugin/xpupti/XpuptiProfilerMacros.h": "9818f9235b391146ad43e3d57b2127a1fdc8ed21b60a00408731bb98d84a5364",
    "src/plugin/xpupti/XpuptiScopeProfilerApi.cpp": "2f200d6b5b6064291c45733a6182f8db13d05de2f5edadacf28283d1fe3a9fc2",
    "src/plugin/xpupti/XpuptiScopeProfilerApi.h": "295e3a2305cc5e2c3f14eabbb84932948bf116eccdea5ef2a3d14bcde9408d33",
    "src/plugin/xpupti/XpuptiScopeProfilerConfig.cpp": "087ac620300a0dc6b6787bad9974adb7db3b1f6af5b1d0e5e7f74faa704394ac",
    "src/plugin/xpupti/XpuptiScopeProfilerConfig.h": "c3cb9bec3a2177ce050635861b1a123799f872facb85f1e517fe0eb9253d2504",
    "src/plugin/xpupti/XpuptiScopeProfilerHandlers.cpp": "9dc67f4a9e5d2556d5f80e849cd8540b4dac2a28bf01bdf0784c5156ad78edc8",
    "src/plugin/xpupti/XpuptiScopeProfilerSession.cpp": "ef23b3322ab043502e5224284adcff93cf1257b2b689468e9c19a1f66fb73dfd",
    "src/plugin/xpupti/XpuptiScopeProfilerSession.h": "b634611047f6358b7e7c0e84b8d0c8e5de09fd50ae65484cc968d929649bfcf9",
}
# END_GENERATED

_URL = "https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{subdir}/{path}"

def _libkineto_repository_impl(rctx):
    # GitHub has no archive of a single subdirectory, and the whole pytorch
    # archive is ~80 MB, so each file the build reads is downloaded on its own.
    # Pinning every file by sha256 keeps them in the repository cache and on
    # the remote downloader, exactly like an http_archive.
    pending = [
        rctx.download(
            url = _URL.format(
                commit = LIBKINETO_COMMIT,
                subdir = LIBKINETO_SUBDIR,
                path = path,
            ),
            output = "libkineto/" + path,
            sha256 = sha256,
            block = False,
        )
        for path, sha256 in LIBKINETO_FILES.items()
    ]
    for download in pending:
        download.wait()

    for patch in rctx.attr.patches:
        rctx.patch(patch, strip = 1)

    rctx.file("BUILD.bazel", rctx.read(rctx.attr.build_file))
    return rctx.repo_metadata(reproducible = True)

libkineto_repository = repository_rule(
    implementation = _libkineto_repository_impl,
    attrs = {
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "patches": attr.label_list(allow_files = True),
    },
    doc = "libkineto's include/, src/ and libkineto_defs.bzl, laid out under " +
          "libkineto/ at the repository root.",
)
