import React from 'react';

import onlineIcon from '../../icons/onlineIcon.png';

import './TextContainer.css';

const TextContainer = ({ users }) => (
  <div className="textContainer">
    <div>
      <img src="https://lh3.googleusercontent.com/d/1nQsZ_SHHwWj3DgFPBcuIxsCp0pZqM_F0=s220" height="75" width="75" class="center"></img>
      <h1>ECE Chat.ML</h1>
      <h2>Developers: Kevin and Devin</h2>
    </div>
    {
      users
        ? (
          <div>
            <h1>Chat Room:</h1>
            <div className="activeContainer">
              <h2>
                {users.map(({name}) => (
                  <div key={name} className="activeItem">
                    {name}
                    <img alt="Online Icon" src={onlineIcon}/>
                  </div>
                ))}
              </h2>
            </div>
          </div>
        )
        : null
    }
  </div>
);

export default TextContainer;