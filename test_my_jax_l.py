import jax
import optax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import vmap,hessian,grad
from tqdm import trange
from jax.experimental import sparse
def init_params(layers):
    params=jnp.ones(0)
    l=[]
    l.append(len(params))
    keys_means = jax.random.split(jax.random.PRNGKey(0),len(layers))
    for keys_mean,shapes in zip(keys_means,layers):
        n_in=shapes[0]
        n_out=shapes[1]
        W_fwd_mean=jax.random.normal(keys_mean,shape=(n_out,n_in))*jnp.sqrt(2/(n_out*n_in))
        B_fwd_mean=jnp.zeros((n_out))
        W_fwd_mean=W_fwd_mean.reshape(n_out*n_in)
        params=np.concatenate((params,W_fwd_mean))
        l.append(len(params))
        B_fwd_mean=B_fwd_mean.reshape(-1)
        params=np.concatenate((params,B_fwd_mean))
        l.append(len(params))
    return params,l

layers=((2,100),
        (100,100),
        (100,100),
        (100,1))

params,l=init_params(layers)
m_par=len(params)
indexes=np.array(l)
indexes=np.concatenate((indexes[:-1].reshape(-1,1),indexes[1:].reshape(-1,1)),axis=1)
indexes=indexes.reshape(-1,2,2)

prior_mean=jnp.zeros_like(params)
prior_std=jnp.ones_like(params)



def model(x,parameter):
    y_mean=x
    for i in range(len(layers)):
        shapes=layers[i]
        n_in=shapes[0]
        n_out=shapes[1]
        W_fwd_mean=parameter[indexes[i,0,0]:indexes[i,0,1]].reshape(n_out,n_in)
        B_fwd_mean=parameter[indexes[i,1,0]:indexes[i,1,1]].reshape(n_out)
        y_mean=W_fwd_mean@y_mean
        y_mean=y_mean+B_fwd_mean

        if i!=(len(layers)-1):
            y_mean=jax.nn.gelu(y_mean)
    return x[0]*(1-x[0])*(x[1])*(1-x[1])+x[0]*(1-x[0])*(x[1])*(1-x[1])*y_mean



data=np.random.rand(100,2)
vmap(model,(0,None))(data,params)

def laplacian(data,params):
    tmp=vmap(jax.hessian(model),(0,None))(data,params)
    return tmp[:,0,0,0]+tmp[:,0,1,1]


eps=0.01
square_inner=eps+(1-2*eps)*np.random.rand(10000,2)
true_sol=(1-square_inner[:,0])*square_inner[:,0]*(1-square_inner[:,1])*square_inner[:,1]
fun=-2*(1-square_inner[:,1])*square_inner[:,1]-2*(1-square_inner[:,0])*square_inner[:,0]


def compute_loss(params,data):
    return jnp.linalg.norm(laplacian(data,params)-fun.reshape(-1))


optim = optax.adam(0.0005)
opt_state=optim.init(params)
loss_and_grad=jax.jit(jax.value_and_grad(compute_loss,argnums=0))

for i in trange(30):
    tot_grad=0
    tot_loss=0
    dem=0
    loss,mygrad=loss_and_grad(params,square_inner)
    updates, opt_state = optim.update(mygrad, opt_state, params)
    params = optax.apply_updates(params, updates)
    print(jnp.linalg.norm(jax.vmap(model,(0,None))(square_inner,params).reshape(-1)-true_sol.reshape(-1))/jnp.linalg.norm(true_sol))

