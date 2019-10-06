// RUN: mlir-opt -split-input-file -simplify-affine-structures %s | FileCheck %s

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (d0 mod 4)
#map = (d0) -> (d0 - (d0 floordiv 4) * 4)

// CHECK: func @f0(memref<100xi8, #map{{[0-9]+}}>)
func @f0(memref<100xi8, #map>)

// -----

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> ((d0 + 1) mod 4)
#map = (d0) -> (d0  - ( (d0 + 1) floordiv 4) * 4 + 1)

// CHECK: func @f1(memref<100xi8, #map{{[0-9]+}}>)
func @f1(memref<100xi8, #map>)

// -----

// CHECK-DAG: #map{{[0-9]+}} = (d0, d1) -> ((d0 + d1 * 2) mod 4)
#map = (d0, d1) -> (d1 + d0  - ( (d0 + 2 * d1) floordiv 4) * 4 + d1)

// CHECK: func @f2(memref<100xi8, #map{{[0-9]+}}>)
func @f2(memref<100x100xi8, #map>)

// -----

// CHECK-DAG: #map{{[0-9]+}} = (d0)[s0] -> ((d0 + 2 * s0) mod 2)
#map = (d0)[s0] -> (d0 + 2 * s0  - ( (d0 + 2 * s0) floordiv 2) * 2)

// CHECK: func @f3(memref<100xi8, #map{{[0-9]+}}>)
func @f3(memref<100xi8, #map>)

// -----

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (d0 mod 8)
#map = (d0) -> (- (d0 floordiv 8) * 8 + d0)

// CHECK: func @f4(memref<100xi8, #map{{[0-9]+}}>)
func @f4(memref<100xi8, #map>)

// -----

// CHECK-DAG: #map{{[0-9]+}} = (d0)[s0] -> ((d0 + 1) mod 4 - 2)
#map = (d0) -> (d0  - ( (d0 + 1) floordiv 4) * 4 - 1)

// CHECK: func @f5(memref<100xi8, #map{{[0-9]+}}>)
func @f5(memref<100xi8, #map>)

// -----

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (((d0 + 1) mod 16) floordiv 8)
#map = (d0) -> ((d0 - ((d0 + 1) floordiv 16) * 16 + 1) floordiv 8)

// CHECK: func @f6(memref<100xi8, #map{{[0-9]+}}>)
func @f6(memref<100xi8, #map>)

// -----

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> ((d0 floordiv 16) * -16)

// This should not simplify to anything.
#map = (d0) -> ( -(d0 floordiv 16) * 16)

// CHECK: func @f7(memref<100xi8, #map{{[0-9]+}}>)
func @f7(memref<100xi8, #map>)
