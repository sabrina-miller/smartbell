//
//  ViewController.swift
//  coremldemo
//
//  Created by Emily Moise on 2/23/20.
//  Copyright © 2020 Emily Moise. All rights reserved.
//

import UIKit
import CoreML
import Foundation

class ViewController: UIViewController, UITextFieldDelegate {
    let input2 = try? MLMultiArray(shape: [45], dataType: MLMultiArrayDataType.double)
    
    var datastring: [Double] = [0.712, 0.687, 0.657, 0.688, 0.647, 0.639, 0.682, 0.666,0.585,0.573,0.604,0.582,0.589,0.635,0.635,0.668,0.662,0.668,0.737,0.752,0.685,0.633,0.647,0.629,0.541,0.543,0.634,0.67,0.679,0.796,-0.004,-0.004,-0.013,-0.007,0.007,-0.014,-0.036,-0.034,-0.013,-0.009,-0.017,-0.014,0,0.011,0.003]
    
    
    @IBOutlet weak var predictionlabel: UILabel!
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    func predictMovement(_ accel:[Double]) -> String? {
        let model = coreml_model()
        
        /*let data = _currentScaledMotionArrays.reduce([], +) //result is of type [Double] with 480 elements*/
        guard let mlMultiArray = try? MLMultiArray(shape:[45], dataType:MLMultiArrayDataType.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        for (index, element) in datastring.enumerated() {
            mlMultiArray[index] = NSNumber(floatLiteral: element)
        }
        let input = coreml_modelInput(accel: mlMultiArray)
        /*guard let predictionOutput = try? _predictionModel.prediction(input: input) else {
                fatalError("Unexpected runtime error. model.prediction")
        }*/
        
        
        
        // Use the generated input API to create the network's input, with no state
       // let modelInput = coreml_modelInput(accel: input2 ?? )
        
        if let prediction = try? model.prediction(input: input) {
            if prediction.classLabel == "Squat" {
                return "Squat"
            }
            else {
                return "Deadlift"
            }
        }
        return nil
    }
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        predictionlabel.text = predictMovement(datastring)
        
        return true
    }
}



