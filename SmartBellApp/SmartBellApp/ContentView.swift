//
//  ContentView.swift
//  SmartBellApp
//
//  Created by Kevin on 2/22/20.
//  Copyright © 2020 SmartBell. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    @State private var selection = 0

    var body: some View {
        TabView(selection: $selection) {
            NavigationView { HomeView() }
                .tabItem {
                    VStack {
                        Image("house")
                        Text("Home")
                    }
                }
                .tag(0)
            NavigationView { WorkoutView() }
                .font(.title)
                .tabItem {
                    VStack {
                        Image("waveform.path.ecg")
                        Text("Workout")
                    }
                }
                .tag(1)
            NavigationView { Text("Logs") }
                .font(.title)
                .tabItem {
                    VStack {
                        Image("doc.plaintext")
                        Text("Logs")
                    }
                }
                .tag(2)
            NavigationView { Text("Settings") }
                .font(.title)
                .tabItem {
                    VStack {
                        Image("gear")
                        Text("Settings")
                    }
                }
                .tag(3)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
