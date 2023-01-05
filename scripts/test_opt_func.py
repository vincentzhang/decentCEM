## this script holds the implementation of the test multi-modal obj function
import matplotlib.pyplot as plt
import numpy as np


def hard1d1(x):
    """ 
        reference: https://machinelearningmastery.com/1d-test-functions-for-function-optimization/
        domain: [-7.5, 7.5]
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-0.888315 at x≈-6.21731
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-0.11909 at x≈-4.1966
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-1.7283 at x≈-2.29609
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-1.48843 at x≈-0.548883
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-0.0135205 at x≈1.39826
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-1.19992 at x≈3.38725
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-1.8996 at x≈5.14574
        min{sin(x) + sin((10 x)/3)|-7.5<=x<=7.5}≈-0.316996 at x≈7.00015

    """
    def objective(x):
        return np.sin(x) + np.sin((10.0 / 3.0) * x)
    return np.array([objective(sample[0]) for sample in x])


if __name__ == "__main__":
    # BEGIN: input arguments
    mode = 1
    render = True
    # End: input arguments
    func_name = {1:"hard1d1"}
    n = 200
    if mode == 1:
        x = np.array([np.linspace(-7.5,7.5,n)])
        x_to_eval = x.reshape(n,1)
        z = hard1d1(x_to_eval) 
    else:
        assert False, "not implemented"

    if render:
        # 3d plot
        fig = plt.figure()
        fontsize = 14
        ax = fig.add_subplot(111)
        ax.plot(x_to_eval, z)
        ax.set_xlabel('x', fontsize=fontsize)
        ax.set_ylabel('f(x)', fontsize=fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.grid(linestyle='--')
        color = ["#ff6eb4", "#ff2500"]
        # for 1d1
        ax.plot([5.1457, 5.1457], [-1.8996, 2], transform=ax.transData, ls='--', color=color[1], alpha=0.5)
        ax.set_ylim([-2.2, 2])
        local_optima = [[-6.217, -.888], [-4.197, -0.119], [-2.296, -1.728], [-0.549, -1.488], 
                       [1.398,-0.014], [3.387, -1.200],  [7, -0.317], [5.146, -1.900]]
        text_font = {'color': '#ff2500', 'size':12}
        for i, val in enumerate(local_optima):
            if i != len(local_optima)-1:
                dot_color = color[0]
                text_color = 'k'
                ax.annotate(r'{}@x={}'.format(val[1], val[0]), xy=(val[0], val[1]), 
                       xytext=(val[0]-0.3, val[1]-0.2), color=text_color,fontsize=fontsize-2 )
            else:
                dot_color = color[1]
                text_color = dot_color
                ax.annotate(r'{}@x*={}'.format(val[1], val[0]), xy=(val[0], val[1]), 
                       xytext=(val[0]-0.3, val[1]-0.2), color=text_color, fontsize=fontsize-2)
            ax.plot([val[0]], [val[1]], 'o', color=dot_color, markersize=4)
        #fig.savefig(f'img/landscape_{func_name[mode]}.pdf', bbox_inches='tight', dpi=600)
        #fig.savefig(f'img/landscape_{func_name[mode]}.png', bbox_inches='tight', dpi=600)

    th = 1.5

    # analyze the global minima
    min_cost = np.min(z)
    print(f'min of the cost function is {min_cost:.5f}')
    min_indices = np.nonzero(z==min_cost) # find minima
    print(f'min index is {min_indices}') # the min_indices[0] indexes y and min_indices[1] indexes x
    flattened_indices = np.ravel_multi_index(min_indices, z.shape)
    print(f'min flattened index is {flattened_indices}')
    print(f'number of global optima {len(flattened_indices)}')
    print(f'all x values at the global optima from the flattened index: {x_to_eval[flattened_indices]}')
