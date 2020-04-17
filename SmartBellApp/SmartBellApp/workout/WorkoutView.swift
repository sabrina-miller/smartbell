//
//  WorkoutView.swift
//  SmartBellApp
//
//  Created by Kevin on 2/22/20.
//  Copyright © 2020 SmartBell. All rights reserved.
//

import MetaWear
import MetaWearCpp
import SwiftUI

struct WorkoutView: View {
    class Accelerometer: ObservableObject {
        @Published public var device: MetaWear? = nil
        @Published public var connected: Bool = false
        private var userCallback: (MblMwCartesianFloat) -> Void = { _ in }
        
        func isConnected() -> Bool {
            var toReturn = device?.isConnectedAndSetup ?? false
            if toReturn != connected {
                // publish on change
                connected = toReturn
            }
            return toReturn
        }
        
        func hasAccelerometer() -> Bool {
            return mbl_mw_metawearboard_lookup_module(device?.board, MBL_MW_MODULE_ACCELEROMETER) != MODULE_TYPE_NA
        }
        
        public func make_context<T: AnyObject>(_ obj: T) -> UnsafeMutableRawPointer {
            return bridge(obj: obj)
        }
        
        func guardSensor(fun: @escaping () -> Void) {
            guard isConnected() else {
                print("Not connected to device")
                return
            }
            guard hasAccelerometer() else {
                print("No accelerometer")
                return
            }
            
            fun()
        }
        
        func subscribe(fun: @escaping (MblMwCartesianFloat) -> Void) {
            guardSensor {
                self.userCallback = fun
                let board = self.device?.board
                let signal = mbl_mw_acc_get_acceleration_data_signal(board)
                
//                 Blink LED While Subscribed
                var pattern = MblMwLedPattern()
                mbl_mw_led_load_preset_pattern(&pattern, MBL_MW_LED_PRESET_PULSE)
                mbl_mw_led_stop_and_clear(board)
                mbl_mw_led_write_pattern(board, &pattern, MBL_MW_LED_COLOR_GREEN)
                mbl_mw_led_play(board)
                
                // Register a callback that calls user supplied callback
                mbl_mw_datasignal_subscribe(signal, self.make_context(self)) { context, data in
                    // Retrieve sensor object
                    let _self: Accelerometer = bridge(ptr: context!)
                    // Retrieve read data
                    let accel3: MblMwCartesianFloat = data!.pointee.valueAs()
                    // Pass data back to user for processing
                    _self.userCallback(accel3)
                }
                
                mbl_mw_acc_enable_acceleration_sampling(board)
                mbl_mw_acc_start(board)
            }
        }
        
        func unsubscribe() {
            guardSensor {
                let board = self.device?.board
                let signal = mbl_mw_acc_get_acceleration_data_signal(board)
                mbl_mw_led_stop_and_clear(board)
                mbl_mw_acc_stop(board)
                mbl_mw_acc_disable_acceleration_sampling(board)
                mbl_mw_datasignal_unsubscribe(signal)
                // nop
                self.userCallback = { _ in }
            }
        }
    }
    
    struct Workout {
        public var weight: String = ""
        public var isWorkingOut: Bool = false
    }
    
    @ObservedObject private var sensor = Accelerometer()
    @State private var workout = Workout()
    
    func makeHeader() -> some View {
        return DispatchView(
            (NavigationLink(destination: ConnectionView($sensor.device)) { Text("Connect") }, { !self.sensor.isConnected() }),
            (Text(sensor.device?.name ?? ""), { self.sensor.isConnected() })
        )
    }
    
    var body: some View {
        return
            VStack {
                VStack(alignment: .leading, spacing: 15) {
                    TextField("Enter Weight (lbs)", text: $workout.weight)                        
                        .padding()
                        .cornerRadius(20.0)
                        .shadow(radius: 10.0, x: 20, y: 10)
                        .keyboardType(.numberPad)
                     }.padding([.leading, .trailing], 35)
                    DispatchView(
                        (
                            Button(action: {
                                if self.workout.isWorkingOut {
                                    self.sensor.unsubscribe()
                                    self.workout.isWorkingOut = false
                                    
                                } else {
                                    self.sensor.subscribe { print($0) }
                                    self.workout.isWorkingOut = true
                                }
                                
                      }) {
                                Text(!workout.isWorkingOut ? "Start" : "Stop")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .padding()
                                    .frame(width: 300, height: 50)
                                    .background(Color.green)
                                    .cornerRadius(15.0)
                                    .shadow(radius: 10.0, x: 20, y: 10)
                            }.padding(.top, 50).navigationBarHidden(true), { self.sensor.isConnected() }
                        ),
                        (NavigationLink(destination: ConnectionView($sensor.device)) { Text("Connect").font(.headline)
                                .foregroundColor(.white)
                                .padding()
                                .frame(width: 300, height: 50)
                                .background(Color.green)
                                .cornerRadius(15.0)
                                .shadow(radius: 10.0, x: 20, y: 10)
                    }, { !self.sensor.isConnected() })
                    )
        }
               
    }
}

//
// struct WorkoutView_Previews: PreviewProvider {
//    static var previews: some View {
//        WorkoutView()
//    }
// }
