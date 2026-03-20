use ndarray::Array1;
use logp::mutual_information::mi_ksg;

fn main() {
    // Test data
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    
    // Calculate MI
    let mi = mi_ksg(&x, &y, 3).unwrap();
    println!("Mutual Information: {}", mi);
    
    // Check if joint MI is available
    println!("logp crate functions:");
    println!("mi_ksg available: yes");
}
