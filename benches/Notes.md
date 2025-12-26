This folder is a separate workspace to perform benchmarks and profiling on the audiograph library. It also offers a performance comparison with the `dasp_graph` library.

## Performance comparison with DASP graph

The comparison of the two libraries is not easy, as both libraries have different design goals and trade-offs. `audiograph` aims to be flexible and takes care of many graph-related aspects (like summing and routing) while `dasp_graph` requires users to implement these aspects on the processing node level.

The goal of these benchmarks is not to determine which library is "faster", but rather to make sure `audiograph` performs reasonably well compared to `dasp_graph`, despite its additional features and flexibility.

The benches include different graph sizes in a diamond shape topology, with both empty processors (to measure graph overhead) and processors that perform some basic audio processing (to measure overall performance). In case of `dasp_graph`, the "empty processor" still has to perform summing to achieve the same functionality as `audiograph`. Whether this is a fair comparison is debatable.

Test results show that `audiograph` is not far behind `dasp_graph` in terms of performance, despite its additional features. In some cases, `audiograph` may even outperform `dasp_graph` - like in the artificial case of empty processors, where the cache locality of `audiograph`'s single summing buffer seems to provide an advantage. The actual performance difference will heavily depend on the node implementations and graph structure.

To run all benchmarks, navigate to the `benches` folder and run `cargo bench`.
