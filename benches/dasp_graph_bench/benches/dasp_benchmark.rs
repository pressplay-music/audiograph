use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

// DASP imports - uses petgraph 0.5 internally
use dasp_graph::node::Node;
use dasp_graph::{Buffer, NodeData};

struct BenchProcessor;

impl Node for BenchProcessor {
    fn process(&mut self, _inputs: &[dasp_graph::Input], output: &mut [Buffer]) {
        for buffer in output.iter_mut() {
            for sample in buffer.iter_mut() {
                *sample = 0.5; // Same fixed computational work as audiograph
            }
        }
    }
}

type DaspGraph = petgraph::graph::DiGraph<NodeData<BenchProcessor>, ()>;
type DaspProcessor = dasp_graph::Processor<DaspGraph>;

fn create_dasp_graph_setup() -> (DaspGraph, DaspProcessor, petgraph::graph::NodeIndex) {
    let mut graph = DaspGraph::with_capacity(10, 10);
    let processor = DaspProcessor::with_capacity(10);

    let node_data = NodeData::new2(BenchProcessor);
    let node_id = graph.add_node(node_data);

    (graph, processor, node_id)
}

fn bench_dasp_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("dasp_performance");

    let (mut dasp_graph, mut dasp_processor, node_id) = create_dasp_graph_setup();

    group.bench_function("full_graph_64_samples", |b| {
        b.iter(|| {
            dasp_processor.process(black_box(&mut dasp_graph), black_box(node_id));
        })
    });

    group.finish();
}

criterion_group!(benches, bench_dasp_performance);
criterion_main!(benches);
