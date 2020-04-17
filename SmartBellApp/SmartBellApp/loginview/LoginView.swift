//
//  LoginView.swift
//  SmartBellApp
//
//  Created by Kevin on 3/6/20.
//  Copyright © 2020 SmartBell. All rights reserved.
//

import SwiftUI

struct LoginView: View {
    @State private var email = ""
    @State private var password = ""
    @Binding private var loggedIn: Bool

    init(_ li: Binding<Bool>) {
        _loggedIn = li
    }

    var body: some View {
//        NavigationView {
        VStack {
            Text("SmartBell")
            VStack(alignment: .leading, spacing: 15) {
                TextField("Email", text: self.$email)
                    .padding()
                    .cornerRadius(20.0)
                    .shadow(radius: 10.0, x: 20, y: 10)

                SecureField("Password", text: self.$password)
                    .padding()
                    .cornerRadius(20.0)
                    .shadow(radius: 10.0, x: 20, y: 10)
            }.padding([.leading, .trailing], 27.5)
            Button(action: { self.loggedIn = true }) {
                Text("Sign In")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(width: 300, height: 50)
                    .background(Color.green)
                    .cornerRadius(15.0)
                    .shadow(radius: 10.0, x: 20, y: 10)
            }.padding(.top, 50)
//            }
        }
    }
}

// struct LoginView_Previews: PreviewProvider {
//    static var previews: some View {
//        LoginView()
//    }
// }
