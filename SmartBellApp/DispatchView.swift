//
//  DispatchView.swift
//  SmartBellApp
//
//  Created by Kevin on 2/25/20.
//  Copyright © 2020 SmartBell. All rights reserved.
//

import SwiftUI

struct DispatchView: View {
    let views: [(AnyView, () -> Bool)]
    init(_ i: (Any, () -> Bool)...) {
        // TODO: Better error handling. This'll crash at
        //       runtime if all of the parameters aren't Views
        views = i.map { (AnyView(_fromValue: $0.0)!, $0.1) }
    }

    var body: some View {
        let selected = views.filter { $0.1() }.map { $0.0 }
        assert(selected.count == 1)
        return selected[0]
    }
}

// struct DispatchView_Previews: PreviewProvider {
//    static var previews: some View {
//        DispatchView(({() -> Bool in true}, ContentView()))
//    }
// }
