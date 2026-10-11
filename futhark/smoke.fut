import "custom-alg1-opt"

entry smoke_input (m: i64) (d: i64) = mk_input m d

-- ==
-- "Smoke 1024-64" script input { smoke_input 16i64 64i64 }
-- output { 0.0f32 }
-- "Smoke 1024-128" script input { smoke_input 8i64 128i64 }
-- output { 0.0f32 }

entry main [m][d] (Q: [m][d][d]f32)
                 (K: [m*d][d]f32)
                 (V: [m*d][d]f32) : f32 =
  validate Q K V
