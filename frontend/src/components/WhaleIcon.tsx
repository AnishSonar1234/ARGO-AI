import React from "react";

export interface WhaleIconProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  className?: string;
  spout?: boolean;
}

export const WhaleIcon: React.FC<WhaleIconProps> = ({ 
  size = 20, 
  className = "w-5 h-5", 
  spout = true,
  ...props 
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      {...props}
    >
      {/* Water Spout Spray */}
      {spout && (
        <g className="whale-spout" opacity="0.9">
          {/* Center jet */}
          <circle cx="12" cy="2" r="0.9" fill="currentColor" />
          <path
            d="M11.6 3.5C11.6 2.8 12 2.2 12 2.2C12 2.2 12.4 2.8 12.4 3.5C12.4 4.2 11.6 4.8 11.6 4.8"
            stroke="currentColor"
            strokeWidth="0.8"
            strokeLinecap="round"
          />
          {/* Left droplet */}
          <circle cx="9.2" cy="3.2" r="0.7" fill="currentColor" />
          {/* Right droplet */}
          <circle cx="14.8" cy="3.2" r="0.7" fill="currentColor" />
        </g>
      )}

      {/* Whale Body */}
      <path
        d="M20.5 12C20.5 8.2 17.5 5.5 12 5.5C7.2 5.5 3.5 8 2.2 11.8C1.5 13.8 2.2 15.8 4 16.5C7 17.6 11.5 16.8 15 15C17.2 13.9 19.2 13.5 20.8 14.2C21.8 14.6 22.8 13.8 22.5 12.8C22.2 11.8 21.2 11.2 20.5 12Z"
        fill="currentColor"
      />

      {/* Whale Fluke (Tail) */}
      <path
        d="M20.2 13.8C20.8 12.6 22 11.2 23.5 10.2C22.6 11.2 21.8 12.4 21.8 13.5C22.6 14.2 23.6 15 23.8 15.6C22.4 15.2 21 14.5 20.2 13.8Z"
        fill="currentColor"
      />

      {/* Flipper / Fin */}
      <path
        d="M8.2 15.2C7.2 16.8 5.8 18.2 4.5 19C5.8 18.8 7.8 17.8 9.2 16.2L8.2 15.2Z"
        fill="currentColor"
      />

      {/* Whale Eye */}
      <circle cx="5.6" cy="11.4" r="0.9" fill="#030712" />
      <circle cx="5.8" cy="11.2" r="0.3" fill="#ffffff" />
    </svg>
  );
};

export default WhaleIcon;
