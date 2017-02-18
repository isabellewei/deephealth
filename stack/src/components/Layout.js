'use strict';

import React from 'react';
import { Link } from 'react-router';

export default class Layout extends React.Component {
  render() {
    return (
		<div>Text located in Layout.js</div>
		//Place where every other page component goes.
        <div className="app-content">{this.props.children}</div>
        
    );
  }
}
