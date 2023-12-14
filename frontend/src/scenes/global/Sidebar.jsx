import React, { useState } from "react";
import { AppBar, Toolbar, IconButton, Typography, useTheme } from "@mui/material";
import { Link } from "react-router-dom";
import MenuOutlinedIcon from "@mui/icons-material/MenuOutlined";
import { tokens } from "../../theme";

const NavbarItem = ({ title, to, selected, setSelected }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  const handleClick = () => {
    setSelected(title);
  };

  return (
    <IconButton
      component={Link}
      to={to}
      onClick={handleClick}
      color={selected === title ? "primary" : "default"}
      sx={{ color: selected === title ? "#ffeb3b" : "inherit" }}
    >
      <Typography variant="h6">{title}</Typography>
    </IconButton>
  );
};

const Navbar = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [selected, setSelected] = useState("Dashboard");

  return (
    <AppBar position="fixed" sx={{ zIndex: theme.zIndex.drawer + 1, backgroundColor: 'grey' }}>
      <Toolbar>
        <IconButton edge="start" color="inherit" aria-label="menu">
          <MenuOutlinedIcon />
        </IconButton>

        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          DM TOOL
        </Typography>

        <NavbarItem
          title="Dashboard"
          to="/"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 1"
          to="/assignment1"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 2"
          to="/assignment2"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 3"
          to="/assignment3"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 4"
          to="/assignment4"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 5"
          to="/assignment5"
          selected={selected}
          setSelected={setSelected}
        />
         <NavbarItem
          title="Assignment 6"
          to="/assignment6"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 7"
          to="/assignment7"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 72"
          to="/assignment72"
          selected={selected}
          setSelected={setSelected}
        />
        <NavbarItem
          title="Assignment 8"
          to="/assignment8"
          selected={selected}
          setSelected={setSelected}
        />
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
