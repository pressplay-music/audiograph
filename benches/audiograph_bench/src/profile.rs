use audiograph::buffer::{AudioBuffer, FrameSize, MultiChannelBuffer};
use audiograph_bench_lib::create_diamond_graph;

fn main() {
    let layers = 9;
    let buffer_size = 64;

    let (mut graph, total_edges) = create_diamond_graph(layers, buffer_size, false); // full processing
    let input = MultiChannelBuffer::new(2, FrameSize(buffer_size));
    let mut output = MultiChannelBuffer::new(2, FrameSize(buffer_size));

    println!("Profiling audiograph graph with {} edges", total_edges);
    println!("Processing loop started - attach Instruments now!");

    // Run many iterations for profiling
    for i in 0..1_000_000 {
        graph.process(&input, &mut output, FrameSize(buffer_size));

        // Prevent optimization and show progress
        if i % 100_000 == 0 {
            println!(
                "Iteration {}: output[0] = {}",
                i,
                output.channel(0).unwrap()[0]
            );
        }
    }

    println!("Profiling complete!");
}
