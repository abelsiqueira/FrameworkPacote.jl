using FrameworkPacote
using NLPModels
using Test

@testset "Básicos" begin
  nlp = ADNLPModel(
    x -> (x[1] - 1)^2 + (x[2] - 2)^2 / 4,
    zeros(2)
  )
  output = gradiente(nlp)
  @test isapprox(output.solution, [1.0; 2.0], rtol=1e-4)
  @test output.objective < 1e-4
  @test output.dual_feas < 1e-4
end