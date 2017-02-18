'use strict';

import React from 'react';
import { Link } from 'react-router';

export default class Layout extends React.Component {
  render() {
    return (
        <div className="app-content">
			<div>Text from Layout.js.</div>
        	{this.props.children}
        </div>
    );
  }
}
