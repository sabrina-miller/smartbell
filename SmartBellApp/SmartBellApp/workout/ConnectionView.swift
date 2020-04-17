//
//  ConnectionView.swift
//  SmartBellApp
//
//  Created by Kevin on 2/22/20.
//  Copyright © 2020 SmartBell. All rights reserved.
//

import AudioToolbox.AudioServices
import Combine
import CoreBluetooth
import MetaWear
import MetaWearCpp
import SwiftUI

struct ConnectionView: View {
    struct Device: Identifiable {
        var id: UUID
        var device: MetaWear
        init(_ d: MetaWear) {
            self.id = d.peripheral.identifier
            self.device = d
        }
    }
    
    class DeviceScanner: ObservableObject {
        var deviceScanner: MetaWearScanner
        init(_ ds: MetaWearScanner) {
            self.deviceScanner = ds
        }
        
        func startScan() {
            deviceScanner.startScan(allowDuplicates: false, callback: { d in
                // Must update on main thread
                DispatchQueue.main.async {
                    // Append new device to list
                    self.deviceList.append(Device(d))
                }
            })
        }
        
        func stopScan(){
            deviceScanner.stopScan()
        }
        
        @Published var deviceList: [Device] = []
    }
    
    //    https://stackoverflow.com/questions/56513568/ios-swiftui-pop-or-dismiss-view-programmatically
    @Environment(\.presentationMode) var presentationMode: Binding<PresentationMode>
    @ObservedObject var ds = DeviceScanner(MetaWearScanner.shared)
    @Binding var connectedDevice: MetaWear?
    @State private var connectionFailed: Bool = false
    
    init(_ cd: Binding<MetaWear?>) {
        _connectedDevice = cd
        startScanningAsync()
    }
    
    func startScanningAsync() {
        ds.startScan()
    }
    
    func connectToBoard(device: MetaWear, onSuccess: @escaping () -> Void) {
        // Hooray! We found a MetaWear board, so stop scanning for more
        ds.stopScan()
        // Connect to the board we found
        device.connectAndSetup().continueWith { t in
            if let error = t.error {
                // Sorry we couldn't connect
                print(error)
                self.connectionFailed = true
                
                self.startScanningAsync()
            } else {
                // Run user supplied success continuation
                onSuccess()
            }
        }
    }
    
    var body: some View {
        List(ds.deviceList) { device in
            Button(
                device.device.name,
                action: {
                    self.connectToBoard(device: device.device, onSuccess: { DispatchQueue.main.async {
                        self.connectedDevice = Optional.some(device.device)
                        // Go back to previous view
                        self.presentationMode.wrappedValue.dismiss()
                        }
                    })
                }
            ).alert(isPresented: self.$connectionFailed) {
                Alert(title: Text("Connection Failed"),
                      message: Text("Could not connect to device " + device.device.name),
                      dismissButton: .default(Text("OK")))
            }
        }
    }
}

//
// struct ConnectionView_Previews: PreviewProvider {
//    static var previews: some View {
//        return ConnectionView()
//    }
// }
