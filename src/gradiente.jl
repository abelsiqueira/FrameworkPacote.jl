export gradiente

function gradiente(
  nlp::AbstractNLPModel;
  atol = 1e-6,
  rtol = 1e-6,
  max_iter = 1000,
  max_time = 10.0
)
  x = copy(nlp.meta.x0)

  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)

  ϵ = atol + rtol * norm(∇f(x))
  t₀ = time()

  iter = 0
  Δt = time() - t₀
  resolvido = norm(∇f(x)) < ϵ
  cansado = iter > max_iter || Δt > max_time

  status = :unknown

  α = 1.0
  while !(resolvido || cansado)

    x -= α * ∇f(x)
    α *= 0.99

    iter += 1
    Δt = time() - t₀
    resolvido = norm(∇f(x)) < ϵ
    cansado = iter > max_iter || Δt > max_time
  end

  if resolvido
    status = :first_order
  elseif cansado
    if iter > max_iter
      status = :max_iter
    elseif Δt > max_time
      status = :max_time
    end
  end

  return GenericExecutionStats(
    status,
    nlp,
    solution=x,
    objective=f(x),
    dual_feas=norm(∇f(x)),
    elapsed_time=Δt,
    iter=iter
  )
end