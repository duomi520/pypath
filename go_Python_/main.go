package main

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
)

var url = "http://127.0.0.1:5000/"

func main() {
	// f(x) =x^3-2x+1
	log.Println(polyintRoots([]float64{1.0, 0.0, -2.0, 1.0})) // [-1.6180339887498945 1.0000000000000004 0.6180339887498948]
	// x+3y+5z-10=0
	// 2x+5y+z-8=0
	// 2x+3y+8z-3=0
	log.Println(linalgSolve([][]float64{{1, 3, 5, -10}, {2, 5, 1, -8}, {2, 3, 8, -3}})) //[-9.280000000000001 5.16 0.76]
	log.Println(brute("lambda x:x**2+10*np.sin(x)", -10, 10))                           //[-1.306411132812531]
}

//解线性方程组
func linalgSolve(np [][]float64) []float64 {
	var result []float64
	if err := call("linalgSolve", np, &result); err != nil {
		log.Fatalln(err)
		return nil
	}
	return result
}

//多项式的根
func polyintRoots(np []float64) []float64 {
	var result []float64
	if err := call("polyintRoots", np, &result); err != nil {
		log.Fatalln(err)
		return nil
	}
	return result
}

//暴力求解
func brute(lambda string, l, h float64) []float64 {
	type Request struct {
		Lambda    string
		Low, High float64
	}
	data := Request{
		Lambda: lambda,
		Low:    l,
		High:   h,
	}
	var result []float64
	if err := call("brute", data, &result); err != nil {
		log.Fatalln(err)
		return nil
	}
	return result
}

func call(name string, data, v interface{}) error {
	var buf []byte
	buffer := bytes.NewBuffer(buf)
	enc := json.NewEncoder(buffer)
	if err := enc.Encode(data); err != nil {
		return err
	}
	res, err := http.Post(url+name, "application/json", buffer)
	if err != nil {
		res.Body.Close()
		return err
	}
	defer res.Body.Close()
	dec := json.NewDecoder(res.Body)
	if err := dec.Decode(&v); err != nil {
		return err
	}
	return nil
}

//https://www.jianshu.com/p/f3c2105bd06b
//https://www.jianshu.com/p/31757e530144
