import numpy as np
from flask import Flask, jsonify, request
from scipy.optimize import brute
from scipy import linalg

app = Flask(__name__)


@app.route('/polyintRoots', methods=['POST'])
def w_polyintRoots():
    result = request.json
    print(result)
    # 多项式可进行加减乘除运算
    p = np.poly1d(result)
    # 多项式的根
    response = np.roots(p)
    return jsonify(response.tolist())


@app.route('/linalgSolve', methods=['POST'])
def w_linalgSolve():
    result = np.array(request.json)
    print(result)
    a = result[:, :-1]
    b = result[:, result.shape[1]-1]*-1
    response = linalg.solve(a, b)  # 线性方程组
    return jsonify(response.tolist())


@app.route('/brute', methods=['POST'])
def w_brute():
    result = request.json
    print(result)
    # 暴力求解
    response = brute(eval(result['Lambda']), ((
        result['Low'], result['High'], 0.1),))
    return jsonify(response.tolist())


if __name__ == '__main__':
    app.debug = True
    app.run()

# https://www.letiantian.me/jiaocheng/python-flask/%E5%85%A5%E9%97%A8/%E8%8E%B7%E5%8F%96POST%E6%96%B9%E6%B3%95%E4%BC%A0%E9%80%81%E7%9A%84%E6%95%B0%E6%8D%AE.html
