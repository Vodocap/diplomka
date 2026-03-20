use ndarray::Array1;
use logp::mutual_information::mi_ksg;

#[test]
fn test_logp_mi()
{
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let mi = mi_ksg(&x, &y, 3).unwrap();
    println!("logp MI: {}", mi);

    // Compare with our implementation
    use crate::entropy::mi_estimator::estimate_mi_ksg;
    let our_mi = estimate_mi_ksg(&x.as_slice().unwrap(), &y.as_slice().unwrap(), 3);
    println!("our MI: {}", our_mi);
}
