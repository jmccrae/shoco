mod model;
mod gen_model;

fn main() {
    let model = gen_model::GenShocoModel::new(vec![
        "/home/jmccrae/projects/jmccrae/shoco/training_data/dorian_gray.txt",
        "/home/jmccrae/projects/jmccrae/shoco/training_data/metamorphosis.txt",
        "/home/jmccrae/projects/jmccrae/shoco/training_data/pride_and_prejudice.txt"])
        .generate().unwrap();

    println!("{:?}", model);
}


